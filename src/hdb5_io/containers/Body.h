//
// Created by lletourn on 26/02/20.
//

#ifndef HDB5_IO_BODY_H
#define HDB5_IO_BODY_H

#include "MathUtils/Matrix.h"
#include "MathUtils/LookupTable1D.h"
#include "MathUtils/VectorN.h"
#include <vector>
#include <unordered_map>
#include <memory>

#include "Mask.h"
#ifdef MESH_SUPPORT
#include "Mesh.h"
#endif
#include "PoleResidue.h"
#include "MathUtils/Vector2d.h"

namespace hdb5_io {

  // Forward declaration
  class HydrodynamicDataBase;

  class WaveDrift;

  using Matrix33 = mathutils::Matrix33<double>;
  using Matrix66 = mathutils::Matrix66<double>;
  using Matrix66b = mathutils::Matrix66<bool>;

  /**
  * \class Body
  * \brief Class for storing the added mass, radiation damping, excitation components, mesh, hydrostatic stiffness, etc.
  * All components are stored, even null DOF components, resulting in 6 DOF vectors and 6x6 matrices.
  */
  class Body {

   public:

    typedef std::unordered_map<unsigned int, std::shared_ptr<mathutils::LookupTable1D<double, mathutils::Vector6d<double>>>> HDBinterpolator;

    /// Constructor for a body
    /// \param id index of the body in the HDB
    /// \param name name of the body
    /// \param hdb pointer to the HDB
    Body(unsigned int id, const std::string &name, HydrodynamicDataBase *hdb);

    /// Return the name of the BEM body database
    /// \return Name of the BEM body
    std::string GetName() const { return m_name; }

    /// Return the identifier of the BEM Body
    /// \return Identifier
    unsigned int GetID() const { return m_id; }

    //
    // Setters
    //

    /// Define the horizontal position of body in world.
    /// \param position Position of the body in world reference frame (NWU)
    void SetHorizontalPositionInWorld(const mathutils::Vector3d<double> &horizontal_position_in_world_frame);

    /// Define the computation point in the body frame.
    /// \param position Position of the computation point in body reference frame
    void SetComputationPointInBodyFrame(const mathutils::Vector3d<double> &computation_point_in_body_frame);

    /// Define the wave reference point in the body frame.
    /// \param position Position of the wave reference point in body frame
    void SetWaveReferencePointInBodyFrame(const mathutils::Vector2d<double> &wave_reference_point_in_body_frame);

    /// Define the mask on the force components
    /// \param mask Mask on the force components : size : nb_force (6)
    void SetForceMask(const mathutils::Vector6d<bool> &mask);

    /// Set the complex matrix of the diffraction coefficients
    /// \param iangle Index of wave direction
    /// \param diffractionMatrix Complex matrix of the diffraction coefficients (nb_forces (6) x nb_freq)
    void SetDiffraction(unsigned int iangle, const Eigen::MatrixXcd &diffractionMatrix);

    /// Set the complex vector of the diffraction coefficients
    /// \param iangle Index of the wave direction
    /// \param iw Index of the wave frequency
    /// \param diffractionVector Complex vector of the diffraction coefficients (nb_forces (6))
    void SetDiffraction(unsigned int iangle, unsigned int iw, const Eigen::VectorXcd &diffractionVector);

    /// Set the complex matrix of the x-derivative of the diffraction coefficient
    /// \param iangle Inde of the wave direction
    /// \param diffractionMatrix Complex matrix of the x-derivative of the diffraction coefficients (nb_forces (6) x nb_freq)
    void SetXDerivativeDiffraction(unsigned int iangle, const Eigen::MatrixXcd &diffractionXDerivativeMatrix);

    /// Set the complex matrix of the x-derivative of the diffraction coefficient
    /// \param iangle Index of the wave direction
    /// \param iw Index of the wave frequency
    /// \param diffractionVector Complex vector of the x-derivative of the diffraction coefficients (nb_forces (6))
    void SetXDerivativeDiffraction(unsigned int iangle, unsigned int iw, const Eigen::VectorXcd &diffractionXDerivativeVector);

    /// Set the complex matrix of the Froude-Krylov coefficient
    /// \param iangle Index of the wave direction
    /// \param froudeKrylovMatrix Complex matrix of the Froude-Krylov coefficients (nb_forces (6) x nb_freq)
    void SetFroudeKrylov(unsigned int iangle, const Eigen::MatrixXcd &froudeKrylovMatrix);

    /// Set the complex vector of the Froude-Krylov coefficient
    /// \param iangle Inex of the wave direction
    /// \param iw Index of the wave frequency
    /// \param froudeKrylovVector Complex vector of the Froude-Krylov coefficients (nb_forces (6))
    void SetFroudeKrylov(unsigned int iangle, unsigned int iw, const Eigen::VectorXcd &froudeKrylovMatrix);

    /// Set the complex matrix of the x-derivative of the Froude-Krylov coefficient
    /// \param iangle Index of the wave direction
    /// \param froudeKrylovMatrix Complex matrix of the x-derivative of the Froude-Krylov coefficients (nb_forces (6) x nb_freq)
    void SetXDerivativeFroudeKrylov(unsigned int iangle, const Eigen::MatrixXcd &froudeKrylovXDerivativeMatrix);

    /// Set the complex vector of the x-derivative of the Froude-Krylov coefficient
    /// \param iangle Index of the wave direction
    /// \param iw Index of the wave frequency
    /// \param froudeKrylovVector Complex vector of the x-derivative of the Froude-Krylov coefficients  (nb_forces (6))
    void SetXDerivativeFroudeKrylov(unsigned int iangle, unsigned int iw, const Eigen::VectorXcd &froudeKrylovXDerivativeMatrix);

    /// Set the complex matrix of the wave excitation coefficients
    /// \param iangle Index of the wave direction
    /// \param excitationMatrix Complex matrix of the wave excitation coefficients (nb_forces (6) x nb_freq)
    void SetExcitation(unsigned int iangle, const Eigen::MatrixXcd &excitationMatrix);

    /// Set the complex vector of the wave excitation coefficients
    /// \param iangle Index of the wave direction
    /// \param iw Index of the wave frequency
    /// \param excitationVector Complex vector of the wave excitation coefficients (nb_forces (6))
    void SetExcitation(unsigned int iangle, unsigned int iw, const Eigen::VectorXcd &excitationVector);

    /// Set the complex matrix of the x-derivative of the wave excitation coefficients
    /// \param iangle Index of the wave direction
    /// \param excitationMatrix Complex matrix of the x-derivative of the wave excitation coefficients (nb_forces (6) x nb_freq)
    void SetXDerivativeExcitation(unsigned int iangle, const Eigen::MatrixXcd &excitationXDerivativeMatrix);

    /// Set the complex vector of the x-derivative of the wave excitation coefficients
    /// \param iangle Index of the wave direction
    /// \param iw Index of the wave frequency
    /// \param excitationVector Complex vector of the x-derivative of the wave excitation coefficients (nb_forces (6))
    void SetXDerivativeExcitation(unsigned int iangle, unsigned int iw, const Eigen::VectorXcd &excitationXDerivativeVector);

    /// Compute the excitation loads from the diffraction loads and the Froude-Krylov loads.
    void ComputeExcitation();

    /// Compute the x-derivative of the excitation loads from the x-derivative of both the diffraction loads and the Froude-Krylov loads.
    void ComputeXDerivativeExcitation();

    /// Set the infinite added mass of the BEM body with respect to the motion of another BEM body
    /// \param BodyMotion BEM body to which the motion is considered
    /// \param CMInf Infinite added mass matrix (nb_forces (6) x nb_motions (6))
    void SetInfiniteAddedMass(Body *BodyMotion, const Matrix66 &CMInf);

    /// Set the x-derivative of the infinite added mass of the BEM body with respect to the motion of another BEM body
    /// \param BodyMotion BEM body to which the motion is considered
    /// \param CMInf x-derivative of the infinite added mass matrix (nb_forces (6) x nb_motions (6))
    void SetXDerivativeInfiniteAddedMass(Body *BodyMotion, const Matrix66 &XDerivativeCMInf);

    /// Set the zero frequency added mass of the BEM body with respect to the motion of another BEM body
    /// \param BodyMotion BEM body to which the motion is considered
    /// \param CMInf zero frequency added mass matrix (nb_forces (6) x nb_motions (6))
    void SetZeroFreqAddedMass(Body *BodyMotion, const Matrix66 &CMZero);

    /// Set the x-deriative of the zero frequency added mass of the BEM body with respect to the motion of another BEM body
    /// \param BodyMotion BEM body to which the motion is considered
    /// \param CMInf x-derivative of the zero frequency added mass matrix (nb_forces (6) x nb_motions (6))
    void SetXDerivativeZeroFreqAddedMass(Body *BodyMotion, const Matrix66 &XDerivativeCMZero);

    /// Set the radiation mask of the BEM body with respect to the motion of another BEM body
    /// \param BodyMotion BEM body to which the motion is considered
    /// \param mask radiation mask of the BEM body with respect to the motion of another BEM body (nb_forces (6) x nb_motions (6))
//    void SetRadiationMask(Body *BodyMotion, const Matrix66b &mask);
    void SetRadiationMask(Body *BodyMotion, const Matrix66b &mask);

    /// Set the complex matrix of the response amplitude operator
    /// \param iangle Index of the wave direction
    /// \param RAO Complex matrix of the response amplitude operator (nb_dof (6) x nb_freq)
    void SetRAO(unsigned int iangle, const Eigen::MatrixXcd &RAO);

    /// Set the complex vector of the response amplitude operator
    /// \param iangle Index of the wave direction
    /// \param iw Index of the wave frequency
    /// \param RAO Complex vector of the response amplitude operator (nb_dof (6))
    void SetRAO(unsigned int iangle, unsigned int iw, const Eigen::VectorXcd &RAO);

    /// Set an impulse response function, as interpolator from a list of data
    /// \param BodyMotion body at the origin of the motion impulse
    /// \param listData data with format iforce x (imotion, iomega) : std::vector is for the iforce dimension, while
    ///         MatrixXds are of dimensions imotion (from the bodyMotion body) x iomega
    void SetIRF(Body *BodyMotion, const std::string &IRF_type, const std::vector<Eigen::MatrixXd> &listData);

    /// Set the added mass for current body iforce dof, from BodyMotion imotion dof, for a set of frequencies
    /// \param BodyMotion body at the origin of the motion
    /// \param listData data with format iomega x (iforce, imotion) : std::vector is for the frequencies dimension, while
    ///         Matrix66s are of dimensions iforce (for this body) x imotion (from the bodyMotion body)
    void SetAddedMass(Body *BodyMotion, const std::vector<Matrix66> &listData);

    /// Set the x-derivative of the added mass for current body iforce dof, from BodyMotion imotion dof, for a set of frequencies
    /// \param BodyMotion body at the origin of the motion
    /// \param listData data with format iomega x (iforce, imotion) : std::vector is for the frequencies dimension, while
    ///         Matrix66s are of dimensions iforce (for this body) x imotion (from the bodyMotion body)
    void SetXDerivativeAddedMass(Body *BodyMotion, const std::vector<Matrix66> &listData);

    /// Set the radiation damping for current body iforce dof, from BodyMotion imotion dof, for a set of frequencies
    /// \param BodyMotion body at the origin of the motion
    /// \param listData data with format iomega x (iforce, imotion) : std::vector is for the frequencies dimension, while
    ///         Matrix66s are of dimensions iforce (for this body) x imotion (from the bodyMotion body)
    void SetRadiationDamping(Body *BodyMotion, const std::vector<Matrix66> &listData);

    /// Set the x-derivative of the radiation damping for current body iforce dof, from BodyMotion imotion dof, for a set of frequencies
    /// \param BodyMotion body at the origin of the motion
    /// \param listData data with format iomega x (iforce, imotion) : std::vector is for the frequencies dimension, while
    ///         Matrix66s are of dimensions iforce (for this body) x imotion (from the bodyMotion body)
    void SetXDerivativeRadiationDamping(Body *BodyMotion, const std::vector<Matrix66> &listData);

    /// Add added mass data for current body iforce dof, from BodyMotion imotion dof, for a frequency
    /// \param BodyMotion body at the origin of the motion
    /// \param Data data with format (iforce, imotion)
    void AddAddedMass(Body *BodyMotion, const Matrix66 &Data);

    /// Add x-derivative of the added mass data for current body iforce dof, from BodyMotion imotion dof, for a frequency
    /// \param BodyMotion body at the origin of the motion
    /// \param Data data with format (iforce, imotion)
    void AddXDerivativeAddedMass(Body *BodyMotion, const Matrix66 &Data);

    /// Add radiation damping for current body iforce dof, from BodyMotion imotion dof, for a frequency
    /// \param BodyMotion body at the origin of the motion
    /// \param Data data with format (iforce, imotion)
    void AddRadiationDamping(Body *BodyMotion, const Matrix66 &Data);

    /// Add x-derivative of the radiation damping for current body iforce dof, from BodyMotion imotion dof, for a frequency
    /// \param BodyMotion body at the origin of the motion
    /// \param Data data with format (iforce, imotion)
    void AddXDerivativeRadiationDamping(Body *BodyMotion, const Matrix66 &Data);

    /// Set the hydrostatic stiffness Matrix
    /// \param hydrostaticStiffnessMatrix Hydrostatic stiffness matrix
    void SetStiffnessMatrix(const Matrix33 &hydrostaticStiffnessMatrix);

    /// Set the hydrostatic stiffness Matrix
    /// \param hydrostaticStiffnessMatrix Hydrostatic stiffness matrix
    void SetStiffnessMatrix(const Matrix66 &hydrostaticStiffnessMatrix);

    /// Set the body inertia matrix, given at the body COG
    /// \param inertiaMatrix body inertia matrix with format (iforce, imotion)
    void SetInertia(const Matrix66 &inertiaMatrix);

    /// Set the mooring stiffness matrix, computed at the body COG
    /// \param mooringMatrix mooring matrix with format (iforce, imotion)
    void SetMooring(const Matrix66 &mooringMatrix);

    /// Set the linear damping matrix, computed at the body COG
    /// \param linearDampingMatrix linear damping matrix with format (iforce, imotion)
    void SetLinearDamping(const Matrix66 &linearDampingMatrix);

    /// Set the static mean drift
    /// \param waveDrift static mean wave drift pointer
    void SetWaveDrift(const std::shared_ptr<WaveDrift> & waveDrift);

#ifdef MESH_SUPPORT
    /// Load the mesh, from vertices and connectivity
    /// \param vertices vertices container
    /// \param faces connectivity of all faces
    void LoadMesh(const std::vector<mathutils::Vector3d<double>> &vertices, const std::vector<Eigen::VectorXi> &faces);
#endif

    /// Add a vector of modal coefficients for the 6 degrees of freedom of this body, generated by one degree of freedom
    /// of the BodyMotion body
    /// \param BodyMotion body at the origin of the perturbation
    /// \param modalCoefficients modal coefficients
    void AddModalCoefficients(Body *BodyMotion, const std::vector<PoleResidue> &modalCoefficients);

    /// This method clears the modal coefficients of a body.
    void ClearModalCoefficients(Body *BodyMotion);

    //
    // Getters
    //

    /// Check if the body contains RAOs
    /// \return true if the body contains RAO
    bool HasRAO() const;

    /// True if the body contains modal coefficients
    /// \param BodyMotion body at the origin of the motion
    /// \return true if the body contains modal coefficients
    bool HasModal(Body *BodyMotion) const;

    /// True if the body contains impulse response functions
    /// \return true if the body contains impulse response functions
    bool HasIRF() const;
    
    /// True if the body contains the inertia matrix
    /// \return true if the body contains the inertia matrix
    bool HasInertia() const;

    /// True if the body contains the hydrostatic matrix
    /// \return true if the body contains the hydrostatic matrix
    bool HasHydrostatic() const;

    /// True if the body contains a mooring matrix
    /// \return true if the body contains a mooring matrix
    bool HasMooring() const;

    /// True if the body contains a linear damping matrix
    /// \return true if the body contains a linear damping matrix
    bool HasDamping() const;

    /// True if the body contains the zero frequency added mass coefficients
    /// \param BodyMotion body at the origin of the motion
    /// \return true if the body contains the zero frequency added mass coefficients
    bool HasZeroFreqAddedMass(Body *BodyMotion) const;

    /// True if the body contains mean wave drift coefficients
    /// \return true if the body contains mean wave drift coefficients
    bool HasWaveDrift() const;

    /// Return the horizontal position of the body in world reference frame (NWU)
    /// \return position of the body in world reference frame (NWU)
    mathutils::Vector3d<double> GetHorizontalPositionInWorld() const;

    /// Return the computation point position of the body in the body frame.
    /// \return position of the computation point in body frame
    mathutils::Vector3d<double> GetComputationPointInBodyFrame() const;

    /// Return the wave reference point in the body frame.
    /// \return position of the wave reference point  in body frame
    mathutils::Vector2d<double> GetWaveReferencePointInBodyFrame() const;

    /// Return the mask value applied on a specific motion mode
    /// \param iforce Index of force
    /// \return Mask on the force mode
    Mask GetForceMask() const;

    #ifdef MESH_SUPPORT
    /// Get the mesh of the body
    /// \return mesh of the body
    Mesh *GetMesh() const;
    #endif

#ifdef USE_VTK

    /// Visualize the mesh of the body
    void VisualizeMesh() const;

#endif

    /// Get the hydrostatic stiffness of this body
    /// \return 3x3 matrix containing the hydrostatic components (heave, roll, pitch)
    Matrix33 GetHydrostaticStiffnessMatrix() const;

    /// Get the inertia matrix
    /// \return the inertia matrix, expressed at COG
    Matrix66 GetInertiaMatrix() const;

    /// Get the linear mooring matrix
    /// \return the linear mooring matrix, expressed at COG
    Matrix66 GetMooringMatrix() const;

    /// Get the linear damping matrix
    /// \return the linear damping matrix, expressed at COG
    Matrix66 GetDampingMatrix() const;

    /// Get the diffraction components for this body
    /// \param iangle Index of the angle
    /// \return matrix containing the complex diffraction components for the given angle with format (nb_force (6), nb_freq)
    Eigen::MatrixXcd GetDiffraction(unsigned int iangle) const;

    /// Get the diffraction components for this body
    /// \param iangle Index of the angle
    /// \param iforce Index of the force dof
    /// \return matrix containing the complex diffraction components for the given angle with format (n_freq)
    Eigen::VectorXcd GetDiffraction(unsigned int iangle, unsigned int iforce) const;

    /// Get the x-derivative of the diffraction components for this body
    /// \param iangle Index of the angle
    /// \return matrix containing the x-derivative of the diffraction components for the given angle with format (nb_force (6), n_freq)
    Eigen::MatrixXcd GetXDerivativeDiffraction(unsigned int iangle) const;

    /// Get the x-derivative of the diffraction components for this body
    /// \param iangle Index of the angle
    /// \param iforce Index of the force dof
    /// \return matrix containing the x-derivative of the x-derivative of the diffraction component for the given angle with format (n_freq)
    Eigen::VectorXcd GetXDerivativeDiffraction(unsigned int iangle, unsigned int iforce) const;

    /// Get the Froude-Krylov components for this body
    /// \param iangle Index of the angle
    /// \return matrix containing the complex Froude-Krylov components for the given angle with format (n_force (6), n_freq)
    Eigen::MatrixXcd GetFroudeKrylov(unsigned int iangle) const;

    /// Get the Froude-Krylov components for this body
    /// \param iangle Index of the angle
    /// \param iforce Index of the force dof
    /// \return vector containing the complex Froude-Krylov components for the given angle with format (n_freq)
    Eigen::VectorXcd GetFroudeKrylov(unsigned int iangle, unsigned int iforce) const;

    /// Get the x-derivative of the Froude-Krylov components for this body
    /// \param iangle index of the angle
    /// \return matrix containing the x-derivative of the Froude-Krylov component for the given angle with format (nb_force (6), nb_freq)
    Eigen::MatrixXcd GetXDerivativeFroudeKrylov(unsigned int iangle) const;

    /// Get the x-derivative of the Froude-Krylov components for this body
    /// \param iangle index of the angle
    /// \param iforce index of the force dof
    /// \return matrix containing the x-derivative of the Froude-Krylov component for the given angle with format (nb_freq)
    Eigen::VectorXcd GetXDerivativeFroudeKrylov(unsigned int iangle, unsigned int iforce) const;

    /// Get the excitation components for this body
    /// \param iangle index of the angle
    /// \return matrix containing the excitation component for the given angle with format (nb_force (6), nb_freq)
    Eigen::MatrixXcd GetExcitation(unsigned int iangle) const;

    /// Get the excitation components for this body
    /// \param iangle index of the angle
    /// \param iforce index of the force dof
    /// \return matrix containing the excitation component for the given angle with format (nb_freq)
    Eigen::VectorXcd GetExcitation(unsigned int iangle, unsigned int iforce) const;

    /// Get the x-derivative of the excitation components for this body
    /// \param iangle index of the angle
    /// \return matrix containing the x-derivative of the excitation component for the given angle with format (nb_force (6), nb_freq)
    Eigen::MatrixXcd GetXDerivativeExcitation(unsigned int iangle) const;

    /// Get the x-derivative of the excitation components for this body
    /// \param iangle index of the angle
    /// \param iforce index of the force dof
    /// \return matrix containing the x-derivative excitation component for the given angle with format (nb_freq)
    Eigen::VectorXcd GetXDerivativeExcitation(unsigned int iangle, unsigned int iforce) const;

    /// Get the infinite added mass, resulting from a motion of body BodyMotion
    /// \param BodyMotion body which motion create added mass on this body
    /// \return 6x6 matrix added mass
    Matrix66 GetInfiniteAddedMass(Body *BodyMotion) const;

    /// Get the x-derivative of the infinite added mass, resulting from a motion of body BodyMotion
    /// \param BodyMotion body which motion create added mass on this body
    /// \return 6x6 matrix added mass
    Matrix66 GetXDerivativeInfiniteAddedMass(Body *BodyMotion) const;

    /// Get the zero frequency added mass, resulting from a motion of body BodyMotion
    /// \param BodyMotion body which motion create added mass on this body
    /// \return 6x6 matrix added mass
    Matrix66 GetZeroFreqAddedMass(Body *BodyMotion) const;

    /// Get the x-derivative of the zero frequency added mass, resulting from a motion of body BodyMotion
    /// \param BodyMotion body which motion create added mass on this body
    /// \return 6x6 matrix added mass
    Matrix66 GetXDerivativeZeroFreqAddedMass(Body *BodyMotion) const;

    /// Get the radiation mask, between this body and BodyMotion body
    /// \param BodyMotion body
    /// \return radiation mask
    Matrix66b GetRadiationMask(Body *BodyMotion) const;

    /// Get the response amplitude operator for this body
    /// \param iangle index of the angle
    /// \return matrix containing the response amplitude operator for the given angle with format (nb_dof (6), nb_freq)
    Eigen::MatrixXcd GetRAO(unsigned int iangle) const;

    /// Get the response amplitude operator for this body
    /// \param iangle index of the angle
    /// \param imotion index of the dof of motion
    /// \return matrix containing the response amplitude operator for the given angle and dof with format (nb_freq)
    Eigen::VectorXcd GetRAO(unsigned int iangle, unsigned int iforce) const;

    /// Get the infinite added mass, resulting from a motion of this body
    /// \return 6x6 matrix added mass
    Matrix66 GetSelfInfiniteAddedMass();

    /// Get the x-derivative infinite added mass, resulting from a motion of this body
    /// \return 6x6 matrix added mass
    Matrix66 GetSelfXDerivativeInfiniteAddedMass();

    /// Get the zero frequency added mass, resulting from a motion of this body
    /// \return 6x6 matrix added mass
    Matrix66 GetSelfZeroAddedMass();

    /// Get the x-derivative zero frequency added mass, resulting from a motion of this body
    /// \return 6x6 matrix added mass
    Matrix66 GetSelfXDerivativeZeroAddedMass();

    /// Get the added mass generated coefficients from the BodyMotion, for this body iforce dof 
    /// \param BodyMotion body at the origin of the motion
    /// \param imotion index of the dof motion
    /// \return matrix containing the added mass with format (nb_dof (6), nb_freq)
    Eigen::MatrixXd GetAddedMass(Body* BodyMotion, unsigned int imotion) const;

    /// Get the x-derivative of the added mass generated coefficients from the BodyMotion, for this body iforce dof
    /// \param BodyMotion body at the origin of the motion
    /// \param imotion index of the dof motion
    /// \return matrix containing the added mass with format (nb_dof (6), nb_freq)
    Eigen::MatrixXd GetXDerivativeAddedMass(Body* BodyMotion, unsigned int imotion) const;

    /// Get the added mass generated coefficients from the BodyMotion, for a given frequency
    /// \param BodyMotion body at the origin of the motion
    /// \param iomega frequency
    /// \return matrix containing the added mass with format (nb_dof (6), nb_motion (6))
    Matrix66 GetAddedMassPerFrequency(Body* BodyMotion, unsigned int iomega) const;

    /// Get the x-derivative added mass generated coefficients from the BodyMotion, for a given frequency
    /// \param BodyMotion body at the origin of the motion
    /// \param iomega frequency
    /// \return matrix containing the added mass with format (nb_dof (6), nb_motion (6))
    Matrix66 GetXDerivativeAddedMassPerFrequency(Body* BodyMotion, unsigned int iomega) const;

    /// Get the radiation damping generated coefficients from the BodyMotion, for this body iforce dof 
    /// \param BodyMotion body at the origin of the motion
    /// \param iforce this body dof
    /// \return matrix containing the radiation damping with format (nb_dof (6), nb_freq)
    Eigen::MatrixXd GetRadiationDamping(Body* BodyMotion, unsigned int imotion) const;

    /// Get the x-derivative of the radiation damping generated coefficients from the BodyMotion, for this body iforce dof
    /// \param BodyMotion body at the origin of the motion
    /// \param iforce this body dof
    /// \return matrix containing the radiation damping with format (nb_dof (6), nb_freq)
    Eigen::MatrixXd GetXDerivativeRadiationDamping(Body* BodyMotion, unsigned int imotion) const;

    /// Get the radiation damping generated coefficients from the BodyMotion, for a given frequency
    /// \param BodyMotion body at the origin of the motion
    /// \param iomega frequency
    /// \return matrix containing the radiation damping with format (nb_force (6), nb_motion (6))
    Matrix66 GetRadiationDampingPerFrequency(Body* BodyMotion, unsigned int iomega) const;

    /// Get the x-derivative of the radiation damping generated coefficients from the BodyMotion, for a given frequency
    /// \param BodyMotion body at the origin of the motion
    /// \param iomega frequency
    /// \return matrix containing the radiation damping with format (nb_force (6), nb_motion (6))
    Matrix66 GetXDerivativeRadiationDampingPerFrequency(Body* BodyMotion, unsigned int iomega) const;

    /// Get an impulse response function interpolator
    /// \return impulse response function interpolator
    HDBinterpolator *GetIRFInterpolator(const std::string& IRF_type) const;

    /// Get an impulse response function interpolated data for the following parameters
    /// \param BodyMotion body at the origin of the perturbation
    /// \param idof index of the degree of freedom considered
    /// \param frequencies set of frequencies for which the data are interpolated
    /// \return interpolated data in matrix with complex components with format (6, nb_freq)
    Eigen::MatrixXd
    GetIRFInterpolatedData(Body *BodyMotion, const std::string& IRF_type, unsigned int idof,
                           mathutils::VectorN<double> frequencies);

    /// Get the modal coefficients (poles and residues) for the 6DOFs
    /// \param BodyMotion body at the origin of the perturbation
    /// \param idof index of the degree of freedom at the origin of the perturbation
    /// \return vector of modal coefficients
    std::vector<PoleResidue> GetModalCoefficients(Body *BodyMotion, int idof);

    /// Get the modal coefficients (poles and residues) for the iforce dof, generated by the idof dof of BodyMotion body
    /// \param BodyMotion body at the origin of the perturbation
    /// \param idof index of the degree of freedom at the origin of the perturbation
    /// \param iforce index of the degree of freedom considered
    /// \return modal coefficients
    PoleResidue GetModalCoefficients(Body *BodyMotion, int idof, int iforce);

    /// Get the mean wave drift model applied to the body
    /// \return mean wave drift
    WaveDrift* GetWaveDrift() const;

   protected:

    HydrodynamicDataBase *m_HDB;                   ///< HDB containing this data container
    unsigned int m_id;                             ///< ID of the BEM Body
    std::string m_name;                            ///< Name of the body
    mathutils::Vector3d<double> m_horizontal_position_in_world_frame;   ///< Horizontal position of the body in the world frame (x, y, psi).
    mathutils::Vector3d<double> m_computation_point_in_body_frame;   ///< Computation point in body frame (m).
    mathutils::Vector2d<double> m_wave_reference_point_in_body_frame; ///< Wave reference point in the body frame (m).

    Mask m_forceMask;                              ///< Mask applied on the force

#ifdef MESH_SUPPORT
    std::shared_ptr<Mesh> m_mesh;                  ///< mesh of the body
#endif

    // TODO :replace treplace Eigen with MathUtils

    /// Excitation loads for all wave directions (std::vector), all dof (rows of Eigen::MatrixXcd) and all wave frequencies (cols of Eigen::MatrixXcd).
    std::vector<Eigen::MatrixXcd> m_excitation;

    /// x-derivative of the excitation loads for all wave directions (std::vector), all dof (rows of Eigen::MatrixXcd) and all wave frequencies (cols of Eigen::MatrixXcd).
    std::vector<Eigen::MatrixXcd> m_excitation_x_derivative;

    /// Froude-Krylov loads for all wave directions (std::vector), all dof (rows of Eigen::MatrixXcd) and all wave frequencies (cols of Eigen::MatrixXcd).
    std::vector<Eigen::MatrixXcd> m_froudeKrylov;

    /// x-derivative of the Froude-Krylov loads for all wave directions (std::vector), all dof (rows of Eigen::MatrixXcd) and all wave frequencies (cols of Eigen::MatrixXcd).
    std::vector<Eigen::MatrixXcd> m_froudeKrylov_x_derivative;

    /// Diffraction loads for all wave directions (std::vector), all dof (rows of Eigen::MatrixXcd) and all wave frequencies (cols of Eigen::MatrixXcd).
    std::vector<Eigen::MatrixXcd> m_diffraction;

    /// x-derivative of the diffraction loads for all wave directions (std::vector), all dof (rows of Eigen::MatrixXcd) and all wave frequencies (cols of Eigen::MatrixXcd).
    std::vector<Eigen::MatrixXcd> m_diffraction_x_derivative;

    bool m_isRAO = false;

    /// Response Amplitude Operators for all wave directions (std::vector), all dof (rows of Eigen::MatrixXcd) and all wave frequencies (cols of Eigen::MatrixXcd).
    std::vector<Eigen::MatrixXcd> m_RAO;

    bool m_isHydrostatic = false;
    Matrix33 m_hydrostaticStiffnessMatrix;         ///< Hydrostatic matrix
    bool m_isInertia = false;
    Matrix66 m_inertia;                            ///< Inertia matrix
    bool m_isMooring = false;
    Matrix66 m_mooringStiffnessMatrix;             ///< Mooring stiffness matrix
    bool m_isDamping = false;
    Matrix66 m_linearDampingMatrix;                ///< Linear damping matrix (not radiation).

    std::unordered_map<Body *, Matrix66b> m_radiationMask;       ///< Radiation mask

    // Infinite frequency added mass.
    std::unordered_map<Body *, Matrix66> m_infiniteAddedMass;

    // x-derivative of the infinite frequency added mass.
    std::unordered_map<Body *, Matrix66> m_infiniteAddedMass_x_derivative;

    // Zero frequency added mass.
    std::unordered_map<Body *, Matrix66> m_zeroFreqAddedMass;

    // x-derivative of the zero frequency added mass.
    std::unordered_map<Body *, Matrix66> m_zeroFreqAddedMass_x_derivative;

    /// Added mass.
    std::unordered_map<Body *, std::vector<Matrix66>> m_addedMass;

    // x-deriative of the added mass.
    std::unordered_map<Body *, std::vector<Matrix66>> m_addedMass_x_derivative;

    // Radiation damping.
    std::unordered_map<Body *, std::vector<Matrix66>> m_radiationDamping;

    // X-derivative of the radiation damping.
    std::unordered_map<Body *, std::vector<Matrix66>> m_radiationDamping_x_damping;

    std::unordered_map<Body *, std::vector<std::vector<PoleResidue>>> m_modalCoefficients;  ///< modal coefficients

    /// Impulse response function interpolators (K, KU, KUXderivative, KU2).
    std::unordered_map<std::string, std::shared_ptr<HDBinterpolator>> m_interpIRF;
    bool m_isIRF = false;

    std::shared_ptr<WaveDrift> m_waveDrift;            ///< wave drift components
    bool m_isWaveDrift = false;

    /// Allocate the body containers
    void AllocateAll();

  };

} // namespace hdb5_io

#endif //HDB5_IO_BODY_H
