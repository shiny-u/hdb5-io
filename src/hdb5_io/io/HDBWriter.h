//
// Created by lletourn on 15/04/20.
//

#ifndef HDB5_IO_HDBWRITER_H
#define HDB5_IO_HDBWRITER_H

#include <string>

#include <highfive/H5File.hpp>
#include "hdb5_io/io/common.h"

namespace hdb5_io {

  // Forward Declarations
  class HydrodynamicDataBase;

  class Body;

  /**
  * \class HDBWriter
  * \brief Class for writing a hydrodynamic database in a .hdb5 file. 
  */
  class HDBWriter {

   public:

    /// Constructor of the HDBWriter
    /// \param hdb hydrodynamic database to write the data
    explicit HDBWriter(HydrodynamicDataBase *hdb) : m_hdb(hdb) {}

    /// Write the hydrodynamic data in a .hdb5 file
    /// \param filename name of the file to write the hydrodynamic data
    virtual void Write(const std::string &filename) const;

   protected:

    enum excitationType {
      Diffraction, DiffractionXDerivative, Froude_Krylov, Froude_KrylovXDerivative
    };

    HydrodynamicDataBase *m_hdb;  ///< hydrodynamic database containing the data

    /// Write basic information contained in the hydrodynamic database (version, date of creation, solver, etc.)
    /// \param file file to export the hydrodynamic database
    virtual void WriteHDBBasics(HighFive::File &file) const;

    /// Write the wave direction, frequency, and time discretizations
    /// \param file file to export the hydrodynamic database
    virtual void WriteDiscretizations(HighFive::File &file) const;

    /// Write the symmetries.
    /// \param file file to export the hydrodynamic database
    virtual void WriteSymmetries(HighFive::File &file) const;

    /// Write basic information related to the body given in the path
    /// \param file file to export the hydrodynamic database
    /// \param path path to the body data in the hdb5
    virtual void WriteBodyBasics(HighFive::File &file, const std::string &path, Body *body) const;

#ifdef MESH_SUPPORT
    /// Write the mesh contained in the hydrodynamic database
    /// \param file file to export the hydrodynamic database
    /// \param path path to the mesh in the file
    /// \param body body containing the mesh
    virtual void WriteMesh(HighFive::File &file, const std::string &path, Body *body) const;
#endif

    /// Write the excitation components
    /// \param type excitation type (Diffraction or Froude_Krylov
    /// \param file file to export the hydrodynamic database
    /// \param path path to the components in the file
    /// \param body body containing the components
    virtual void
    WriteExcitation(excitationType type, HighFive::File &file, const std::string &path, Body *body) const;

    /// Write the radiation components
    /// \param file file to export the hydrodynamic database
    /// \param path path to the components in the file
    /// \param body body containing the components
    virtual void WriteRadiation(HighFive::File &file, const std::string &path, Body *body) const;

    /// Write the response amplitude operators
    /// \param file file to export the hydrodynamic database
    /// \param path path to the components in the file
    /// \param body body containing the components
    virtual void WriteRAO(HighFive::File &file, const std::string &path, Body *body) const;

    /// Write the wave drift data
    /// \param file file to export the hydrodynamic database
    virtual void WriteWaveDrift(HighFive::File &file, const std::string &path, Body *body) const;

    virtual void WriteKochin(HighFive::File& HDF5_file) const;

    /// Write the wave field data..
    /// \param file file to export the hydrodynamic database
    virtual void WriteWaveField(HighFive::File &file) const;

    /// Write the vector fitting data
    /// \param file file to export the hydrodynamic database
    virtual void WriteVF(HighFive::File &file) const;

    /// This method writes the expert numerical parameters.
    virtual void WriteExpertParameters(HighFive::File &file) const;

  };

  /// Export the hydrodynamic database given, in a .hdb5 file format
  /// \param filename name of the file to export the hydrodynamic database
  /// \param hdb hydrodynamic database containing the data
  void export_HDB(const std::string &filename, HydrodynamicDataBase *hdb);

}

#endif //HDB5_IO_HDBWRITER_H
