def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    # Write code here
    assignments = []
    for point in points:
        min_dist = float('inf')
        nearest = -1
        for i, centroid in enumerate(centroids):
            dist = sum((p - c) ** 2 for p, c in zip(point, centroid))
            if dist < min_dist:
                min_dist = dist
                nearest = i

        assignments.append(nearest)
    return assignments