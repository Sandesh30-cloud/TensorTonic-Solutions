def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    recommended_k = recommended[:k]
    hit = len(set(recommended_k) & set(relevant))
    if k > 0:
        precision = hit/k
    else :
        precision = 0
    if len(relevant) > 0:
        recall = hit/len(relevant)
    else:
        recall = 0

    return [precision, recall]