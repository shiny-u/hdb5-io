//
// Created by lletourn on 06/04/20.
//

#ifndef HDB5_IO_MESH_H
#define HDB5_IO_MESH_H

#include "meshoui/mesh.h"
#include "MathUtils/Matrix.h"

namespace hdb5_io {

  /**
  * \class Mesh
  * \brief Class for storing the mesh of a body, from a meshouid::mesh
  */
  class Mesh : public meshoui::Mesh {

   public:

    /// Get the vertices positions, in a 3xn Eigen matrix
    /// \return 3xn eigen matric containing the vertices positions
    Eigen::MatrixXd GetVertices() const;

    /// Get the connectivity of the faces of the mesh, in a 3xn Eigen matrix
    /// \return  3xn eigen matric containing the faces connectivity
    Eigen::MatrixXi GetFaces();

  };

}

#endif //HDB5_IO_MESH_H
