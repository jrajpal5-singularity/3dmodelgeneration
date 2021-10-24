
import open3d as o3d
import numpy as np


def load_point_clouds(voxel_size=0.0):
    pcds = []
    normals = []
    for i in range(5):
        pcd = o3d.io.read_point_cloud("point_cloud%d.ply" %i)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        normal = pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcds.append(pcd_down)
        normals.append(normal)
    return pcds, normals


def pairwise_registration(source, target):
    print("Apply point-to-plane ICP")
    # target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


def full_registration(pcds, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    """Fuses 5 point clouds.

    This method takes 5 point clouds, register them with multiway registration and combine them into one point cloud .
    according to the determined transformation (we get from the multiway registration.

    :param pcds: contains all 5 point clouds that are meant to be fuse
    :type: Array
    :param max_correspondence_distance_coarse: is the threshold for correspondence for coarse alignment
    :type: float
    :param max_correspondence_distance_fine: is the threshold for correspondence for fine alignment
    :type: float
    """
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id])
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph


def compare_point_clouds_one_by_one(source_pcd, target_pcd):

    dists = source_pcd.compute_point_cloud_distance(target_pcd)
    dists = np.asarray(dists)
    ind = np.where(dists > 0.01)[0]
    compared_pcd = source_pcd.select_by_index(ind)
    for i in ind:
        differences = compared_pcd.paint_uniform_color([1,0.706, 0])
    o3d.visualization.draw_geometries([target_pcd+differences])


if __name__ == "__main__":
    voxel_size = 0.02
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5
    pcds, normals = load_point_clouds(voxel_size)
    # source = o3d.io.read_point_cloud("path/to/point_cloud.pcd")
    # target = o3d.io.read_point_cloud("path/to/point_cloud.pcd")
    # source.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # target.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pose_graph = full_registration(pcds, max_correspondence_distance_coarse, max_correspondence_distance_fine)
    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(pcds)):
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        pcd_combined += pcds[point_id]
    pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
    o3d.io.write_point_cloud("multiway_registration.ply", pcd_combined_down)
    o3d.visualization.draw_geometries([pcd_combined_down])
    # compare_point_clouds_one_by_one(source, target)
