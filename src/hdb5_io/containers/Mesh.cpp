//
// Created by lletourn on 06/04/20.
//

#include "Mesh.h"


namespace hdb5_io {


  Eigen::MatrixXd Mesh::GetVertices() const {

    Eigen::MatrixXd VertexMatrix(n_vertices(), 3);

    unsigned int i = 0;

    for (const auto &vertex : vertices()) {
      VertexMatrix.row(i) = this->point(vertex);
      i++;
    }

    return VertexMatrix;
  }


  Eigen::MatrixXi Mesh::GetFaces() {

    Eigen::MatrixXi VertexMatrix(n_faces(), 3);

    unsigned int i = 0;

    for (const auto &face : faces()) {

      Eigen::Vector3i connectivity;
      int id = 0;
      auto fv_it = fv_iter(face);
      for (; fv_it.is_valid(); fv_it++) {
        connectivity[id] = (*fv_it).idx();
        id++;
      }
      VertexMatrix.row(i) = connectivity;
      i++;
    }

    return VertexMatrix;
  }

}
