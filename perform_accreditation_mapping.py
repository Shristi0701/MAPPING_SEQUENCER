import json
import os
import numpy as np
from mapping.similarity import score_one_pair, preprocess_co, model, get_keywords
from sklearn.metrics.pairwise import cosine_similarity

def calculate_levels(source_texts, target_texts):
    results = []
    # Batch encode source and target
    source_embs = model.encode([preprocess_co(t) for t in source_texts])
    target_embs = model.encode(target_texts)
    
    for i, s_text in enumerate(source_texts):
        s_emb = source_embs[i]
        candidates = []
        for j, t_text in enumerate(target_texts):
            # We skip the SVM classifier since these are PSOs/PEOs (Zero-Shot)
            # Use BERT and Keyword overlap
            bert_sim = float(cosine_similarity([s_emb], [target_embs[j]])[0][0])
            
            # Simplified keyword score for generic comparison
            s_words = set(preprocess_co(s_text).lower().split())
            t_words = set(t_text.lower().split())
            common = s_words & t_words
            kw_score = min(len(common) / 3.0, 1.0)
            
            # Hybrid score (65% BERT, 35% Keyword for zero-shot)
            hybrid = round(0.65 * bert_sim + 0.35 * kw_score, 3)
            
            # Level mapping
            if hybrid < 0.2: level = 0
            elif hybrid < 0.4: level = 1
            elif hybrid < 0.6: level = 2
            else: level = 3
            
            candidates.append({
                "target": t_text,
                "score": hybrid,
                "level": level
            })
        results.append({
            "source": s_text,
            "mappings": candidates
        })
    return results

def main():
    # 1. Load definitions
    with open("data/accreditation_defs.json") as f:
        defs = json.load(f)
    
    # 2. Extract COs from raw_cos.txt
    cos = []
    with open("raw_cos.txt", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("- ") and len(line) > 10:
                cos.append(line[2:])
    
    print(f"Extracted {len(cos)} COs from raw_cos.txt")
    
    # Take a sample of 20 COs for the first report to avoid blowing up memory/output
    sample_cos = cos[:20]
    
    # 3. CO -> PSO Mapping
    print("Mapping CO to PSO...")
    pso_texts = [p["text"] for p in defs["psos"]]
    co_pso_results = calculate_levels(sample_cos, pso_texts)
    
    # 4. PO -> PEO Mapping
    print("Mapping PO to PEO...")
    po_texts = [p["text"] for p in defs["pos"]]
    peo_texts = [p["text"] for p in defs["peos"]]
    po_peo_results = calculate_levels(po_texts, peo_texts)
    
    # 5. Save results
    final_output = {
        "co_to_pso": co_pso_results,
        "po_to_peo": po_peo_results
    }
    
    with open("data/accreditation_mappings.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    
    print("Full results saved to data/accreditation_mappings.json")
    
    # Print a summary for the user
    print("\nSUMMARY OF PSO MAPPING (First 3 COs):")
    for res in co_pso_results[:3]:
        print(f"CO: {res['source'][:60]}...")
        for m in res['mappings']:
            print(f"  -> {m['target'][:40]}...: Level {m['level']} (Score: {m['score']})")

if __name__ == "__main__":
    main()
