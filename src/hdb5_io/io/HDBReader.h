//
// Created by lletourn on 14/04/20.
//

#ifndef HDB5_IO_HDBREADER_H
#define HDB5_IO_HDBREADER_H

#include <string>
#include <Eigen/Dense>

#include <highfive/H5Group.hpp>
#include "hdb5_io/io/common.h"

namespace hdb5_io {

  // Forward Declaration
  class HydrodynamicDataBase;

  class Body;

  /**
  * \class HDBReader
  * \brief Class for reading a hydrodynamic database from a .hdb5 file. 
  */
  class HDBReader {

   public:

    /// Constructor of the HDBReader
    /// \param hdb hydrodynamic database in which store the data
    explicit HDBReader(HydrodynamicDataBase *hdb) : m_hdb(hdb) {}

    /// Read a hdb5 file
    /// \param filename name of the file to read
    virtual void Read(const std::string &filename);

   protected:

    enum excitationType {
      Diffraction, DiffractionXDerivative, Froude_Krylov, Froude_KrylovXDerivative
    };

//    double m_version;

    HydrodynamicDataBase *m_hdb;  ///< hydrodynamic database to store the data

    /// Read basic information contained in the hydrodynamic database (version, date of creation, solver, etc.)
    /// \param file file containing the hydrodynamic database
    virtual void ReadHDBBasics(const HighFive::File &file);

    /// Read basic information related to the body given in the path
    /// \param file file containing the hydrodynamic database
    /// \param path path to the body data in the hdb5
    /// \return container of the body hydrodynamic data
    virtual Body *ReadBodyBasics(const HighFive::File &file, const std::string &path);

    /// Read the wave direction, frequency, and time discretizations 
    /// \param file file containing the hydrodynamic database
    virtual void ReadDiscretizations(const HighFive::File &file);

    /// Read the symmetry data
    /// \param file file containing the hydrodynamic database
    virtual void ReadSymmetries(HighFive::File &file);

    /// Read the excitation components
    /// \param type excitation type (Diffraction or Froude_Krylov
    /// \param file file containing the hydrodynamic database
    /// \param path path to the components in the file
    /// \param body body to which store the components
    virtual void
    ReadExcitation(excitationType type, const HighFive::File &file, const std::string &path, Body *body);

    /// Read the radiation components
    /// \param file file containing the hydrodynamic database
    /// \param path path to the components in the file
    /// \param body body to which store the components
    virtual void ReadRadiation(const HighFive::File &file, const std::string &path, Body *body);

    /// Read the response amplitude operators
    /// \param file file containing the hydrodynamic database
    /// \param path path to the components in the file
    /// \param body body to which store the components
    virtual void ReadRAO(const HighFive::File &file, const std::string &path, Body *body);

    /// Read the components (added mass/radiation damping/ impulse response functions)
    /// \param file file containing the hydrodynamic database
    /// \param path path to the components in the file
    /// \param radiationMask radiation mask : size nb_force (6) x nb_motion (6)
    /// \return vector of size nb_motions (6) of complex matrix components of dimension :
    ///         - nb_forces (6) x nb_freq : for added mass and damping
    ///         - nb_forces (6) x nb_times : for IRF
    virtual std::vector<Eigen::MatrixXd> ReadComponents(const HighFive::File &file, const std::string &path,
                                                        Eigen::Matrix<bool,6,6> radiationMask);

    /// Read the wave drift data
    /// \param file file containing the hydrodynamic database
    virtual void ReadWaveDrift(HighFive::File &file);

    /// Read the mesh contained in the HDF5 file
    /// \param file file containing the mesh
    /// \param path path to the mesh in the file
    /// \param body body to which store the mesh
    virtual void ReadMesh(HighFive::File &file, const std::string &path, Body *body);

    /// Read the wave field data
    /// \param file file containing the hydrodynamic database
    virtual void ReadWaveField(HighFive::File &file);

    /// Read the wave drift data
    /// \param file file containing the hydrodynamic database
    virtual void ReadVectorFitting(HighFive::File &file);

    /// This method reads the expert numerical parameters.
    /// \param file file containing the hydrodynamic database
    virtual void ReadExpertNumericalParameters(HighFive::File &file);

    /// Read the wave drift components, from the path given, for the i-th body
    /// \param file file containing the hydrodynamic database
    /// \param path path in the hdb5 file to wave drift data
    /// \param i index of the wave direction
    /// \return complex vector of wave drift components (size nb_freq)
    virtual Eigen::VectorXd
    ReadWaveDriftComponents(HighFive::File &file, const std::string &path, unsigned int i);

    /// This method reads the expert Kochin parameters.
    /// \param HDF5_file file containing the hydrodynamic database
    virtual void ReadKochin(const HighFive::File& HDF5_file);

  };

  /// Import a hydrodynamic database from a .hdb5 file
  /// \param filename name of the file to import
  /// \return hydrodynamic database
  std::shared_ptr<HydrodynamicDataBase> import_HDB(const std::string &filename);


} // end namespace hdb5_io
#endif //HDB5_IO_HDBREADER_H
