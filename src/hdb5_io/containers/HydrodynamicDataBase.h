//
// Created by lletourn on 26/02/20.
//

#ifndef HDB5_IO_HYDRODYNAMICDATABASE_H
#define HDB5_IO_HYDRODYNAMICDATABASE_H

#include <iostream>
#include <memory>

#include "MathUtils/Vector3d.h"
#include "MathUtils/VectorN.h"

namespace hdb5_io {

  // Forward declarations

  class WaveDrift;
  class Kochin;

  class Body;

  /**
  * \class HydrodynamicDataBase
  * \brief Class for storing a hydrodynamic database. Contains basic information (version, date of creation, solver, ...)
   * and body and wave drift container class containing all related hydrodynamic data.
  */
  class HydrodynamicDataBase {

   public:

    // Accessors

    /// Set the creation date of the hydrodynamic database
    /// \param date Creation date format YYYY-MM-DD HH:mm:ss.ms
    void SetCreationDate(std::string date);

    /// Get the creation date of the hydrodynamic database
    /// \return Creation date format YYYY-MM-DD HH:mm:ss.ms
    std::string GetCreationDate() const;

    /// Set hydrodynamic solver name
    /// \param solver name of the solver
    void SetSolver(std::string solver);

    /// Get the hydrodynamic solver used
    /// \return name of the solver
    std::string GetSolver() const;

    /// Set the version of hdb
    /// \param commit_hash version from commit_hash
    void SetNormalizedVersionString(std::string commit_hash);

    /// Return the version number
    /// \return version
    std::string GetNormalizedVersionString() const;

    /// Return true if normalized version is defined
    /// \return true if normalizer version
    bool IsNormalizedVersionString() const;

    /// Set the version number
    /// \param version version
    void SetVersion(double version);

    /// Get the version number
    /// \return version
    double GetVersion() const;

    /// Set the gravity acceleration
    /// \param g gravity acceleration
    void SetGravityAcceleration(double g);

    /// Get the gravity acceleration
    /// \return gravity acceleration
    double GetGravityAcceleration() const;

    /// Set the water density (kg/m3)
    /// \param rho water density
    void SetWaterDensity(double rho);

    /// Get the water density (kg/m3)
    /// \return water density
    double GetWaterDensity() const;

    /// Set the water depth (m)
    /// \param h water depth
    void SetWaterDepth(double h);

    /// Return the water depth (m)
    /// \return water depth
    double GetWaterDepth() const;

    /// Set the body normalized length (m)
    /// \param L normalized length
    void SetNormalizationLength(double L);

    /// Get the normalized length (m)
    /// \return normalized length
    double GetNormalizationLength() const;

    /// Add a body to the hydrodynamic database
    /// \param id index of the body in the database
    /// \param name name o the body, must be unique
    /// \return body created
    Body *NewBody(unsigned int id, const std::string &name);

    /// Get the body by index
    /// \param id index of the body in database
    /// \return body
    Body *GetBody(int id) const;

    /// Return the body of the given name
    /// \param name name of the body
    /// \return body
    Body *GetBody(const std::string& name) const;

    /// Set the number of bodies in database
    /// \param nb number of bodies
    void SetNbBodies(int nb);

    /// Get the number of bodies in database
    /// \return number of bodies
    int GetNbBodies() const;

    /// Set the wave frequency discretization
    /// \param frequency list of frequencies (rad/s)
    void SetFrequencyDiscretization(const mathutils::VectorN<double> &frequency);

    /// Set the wave direction discretization
    /// \param directions list of wave directions (rad)
    void SetWaveDirectionDiscretization(const mathutils::VectorN<double> &directions);

    /// Set the time discretization
    /// \param time list of discrete time (s)
    void SetTimeDiscretization(const mathutils::VectorN<double> &time);

    /// Return the frequency discretization
    /// \return vector of frequencies (rad/s)
    mathutils::VectorN<double> GetFrequencyDiscretization() const;

    /// Return the wave direction discretization
    /// \return vector of wave directions (rad)
    mathutils::VectorN<double> GetWaveDirectionDiscretization() const;

    /// Return the time discretization
    /// \return vector of discrete time (s)
    mathutils::VectorN<double> GetTimeDiscretization() const;

    /// Return the minimal wave frequency
    /// \return minimal wave frequency (rad/s)
    double GetMinFrequency() const;

    /// Return the maximal wave frequency
    /// \return maximal wave freuqnecy (rad/s)
    double GetMaxFrequency() const;

    /// Set the kochin model
    /// \param pointer to kochin model
    void SetKochin(const std::shared_ptr<Kochin> &kochin);

    /// Get the Kochin model
    /// \return pointer to kochin model
    Kochin* GetKochin() const;

    void SetVF();

    bool GetVF() const;

    void SetVFRelaxed(const int &relaxed);

    int GetVFRelaxed() const;

    void SetVFMaxOrder(const int &order);

    int GetVFMaxOrder() const;

    void SetVFTolerance(const double &tolerance);

    double GetVFTolerance() const;

    /// Activate wave field
    void SetWaveField();

    /// Return true if wave field activate
    /// \return wave field activated
    bool GetWaveField() const;

    /// Activate symmetries
    void SetSymmetries();

    /// Return true if symmetries are activated
    /// \return symmetries activated
    bool GetSymmetries() const;

    /// Activate or deactivate bottom symmetry
    /// \param sym_bottom True if bottom symmetry is activated
    void SetSymBottom(const bool &sym_bottom);

    /// Return true if bottom symmetry is activated
    /// \return bottom symmetry activated
    bool GetSymBottom() const;

    /// Activate or deactivate Oxz symmetry
    /// \param sym_xOz True if Oxz symmetry is activated
    void SetSymXOZ(const bool &sym_xOz);

    /// Return true if Oxz symmetry is activated
    /// \return Oxz symmetry activated
    bool GetSymXOZ() const;

    /// Activate or deactivate Oyz symmetry
    /// \param sym_yOz True if Oyz symmetry is activated
    void SetSymYOZ(const bool &sym_yOz);

    /// Return true if Oyz symmetry is activated
    /// \return Oyz symmetry activated
    bool GetSymYOZ() const;

    /// Activate expert parameters
    void SetExpertParameters();

    /// Return true if expert parameters are activated
    /// \return expert parameters activated
    bool GetExpertParameters() const;

    /// Set the order of surface integration functions
    /// \param order integration order
    void SetSurfaceIntegrationOrder(const int &order);

    /// Return the surface integration order
    /// \param integration order
    int GetSurfaceIntegrationOrder() const;

    /// Set the name of the method used to evaluate Green function
    /// \param green_function name of the method
    void SetGreenFunction(const std::string &green_function);

    /// Return the name of the method used to evaluate Green functions
    /// \return method name
    std::string GetGreenFunction() const;

    /// Set the distance criteria threshold for applying the far-field approximation for computing the influence coefficients.
    /// \param crmax distance criteria (m)
    void SetCrmax(const int &crmax);

    /// Return the distance criteria threshold for applying the far-field approximation for computing the influence coefficients.
    /// \return distance criteria (m)
    int GetCrmax() const;

//    void SetWaveReferencePoint(const double &x, const double &y);

//    void GetWaveReferencePoint(double &x, double &y) const;

    /// Activate presence of x-derivative of the physical quantities
    void IsXDerivative();

    /// Return true if x-derivative of physical quantities are present
    /// \returb x-derivatives present
    bool GetIsXDerivative() const;

   protected:

    std::string m_creationDate;       ///< Creation date of the HDB
    std::string m_solver;             ///< Solver which computed the hydrodynamic data base (NEMOH/HELIOS)
    bool m_is_commit_hash = false;
    std::string m_commit_hash;        ///< Normalized commit hash.

    double m_version;                 ///< Version of the HDB file
    double m_gravityAcceleration;     ///< Gravity coming from the HDB
    double m_waterDensity;            ///< Water density coming from the HDB
    double m_waterDepth;              ///< Water depth coming from the HDB
    double m_normalizationLength;     ///< Normalization length coming from the HDB

    bool m_isVF = false;
    int m_VF_relaxed; /// Relaxed vector fitting (1) or not (0).
    int m_VF_max_order; /// Maximum order of the vector fitting.
    double m_VF_tolerance; /// Tolerance of the vector fitting.

    bool m_isWaveField = false;

    bool m_isSymmetries = false;
    bool m_sym_bottom; /// Bottom symmetry.
    bool m_sym_xoz; /// (xOz) symmetry.
    bool m_sym_yoz; /// (yOz) symmetry.

    bool m_isExpertParameters = false;
    int m_surface_integration_order; /// Surface integration order.
    std::string m_Green_function; /// Method for evaluating the Green's function.
    double m_Crmax; /// Distance criteria threshold for applying the far-field approximation for computing the influence coefficients.
    double m_wave_reference_point_x; // x-coordinate of the wave reference point (m).
    double m_wave_reference_point_y; // y-coordinate of the wave reference point (m).

    int m_nbody;                      ///< Number of bodies in interaction considered in the HDB
    // FIXME :: change the vector to unordered_map with index as key, to be sure which body to get with the GetBody method
    std::vector<std::shared_ptr<Body>> m_bodies;      ///< List of BEM body database

    mathutils::VectorN<double> m_frequencyDiscretization;
    mathutils::VectorN<double> m_timeDiscretization;
    mathutils::VectorN<double> m_waveDirectionDiscretization;

    std::shared_ptr<Kochin> m_kochin;            ///< Kochin functions.

    bool m_is_x_derivative = false; /// Presence if the x-derivative of the physical quantities.

  };

} // namespace hdb5_io

#endif //HDB5_IO_HYDRODYNAMICDATABASE_H
