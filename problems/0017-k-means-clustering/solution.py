
def k_means_clustering(points: list[tuple[float, ...]], k: int,
                       initial_centroids: list[tuple[float, ...]],
                       max_iterations: int) -> list[tuple[float, ...]]:
    centroids = initial_centroids

    for i in range(max_iterations):
        assignment, centroid_points = assignments(centroids, points)

        new_centroids = []

        for key, value in centroid_points.items():
            if len(value) == 0:
                new_centroids.append(key)
                continue

            new_centroid = tuple(
                sum(point[j] for point in value) / len(value)
                for j in range(len(key))
            )

            new_centroids.append(new_centroid)

        centroids = new_centroids

    return centroids

def distance(point1,point2):
	dist=0
	for i in range(len(point1)):
		dist+=(point1[i]-point2[i])**2
	return dist 

def assignments(considered_centroids, points):
	assignments={}
	centroid_points={}
	for centroid in considered_centroids:
		centroid_points[centroid]=[]
	for point in points:
		min_centroid=None
		min_dist=None
		for centroid in considered_centroids:
			if(min_dist is None or distance(centroid,point)<min_dist):
				min_dist=distance(centroid,point)
				min_centroid=centroid
		assignments[point]=min_centroid
		centroid_points[min_centroid].append(point)
	return assignments, centroid_points