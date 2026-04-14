import numpy as np


def recall_at_k(retrieved_images, ground_truth_images, k):
    relevant_retrieved = [img for img in retrieved_images[:k] if img in ground_truth_images]
    print (relevant_retrieved)
    return len(relevant_retrieved) / len(ground_truth_images)

def recall_at_k_sim(similarity_matrix, k):
    recalls = []
    num_queries = similarity_matrix.shape[0]
    
    for i in range(num_queries):
        sorted_indices = np.argsort(-similarity_matrix[i])
        top_k = sorted_indices[:k]
        
        if i in top_k:
            recalls.append(1)
        else:
            recalls.append(0)
    
    return np.mean(recalls)

def average_precision(similarity_matrix, query_idx):
    sorted_indices = np.argsort(-similarity_matrix[query_idx])
    num_relevant = 1
    ap = 0.0
    hits = 0
    
    for i, idx in enumerate(sorted_indices):
        if idx == query_idx:
            hits += 1
            precision = hits / (i + 1)
            ap += precision
            
    return ap / num_relevant

def mean_average_precision(similarity_matrix):
    aps = []
    num_queries = similarity_matrix.shape[0]
    
    for i in range(num_queries):
        ap = average_precision(similarity_matrix, i)
        aps.append(ap)
    
    return np.mean(aps)
