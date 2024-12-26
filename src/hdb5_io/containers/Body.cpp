//
// Created by lletourn on 26/02/20.
//

#include "Body.h"

#ifdef USE_VTK

#include "meshoui/vtkmesh.h"

#endif

#include "HydrodynamicDataBase.h"


namespace hdb5_io {

  Body::Body(unsigned int id, const std::string &name, HydrodynamicDataBase *hdb) : m_id(id), m_name(name), m_HDB(hdb) {
    #ifdef MESH_SUPPORT
    m_mesh = std::make_shared<Mesh>();
    #endif
    auto interpK = std::make_shared<HDBinterpolator>();
    m_interpIRF.insert(std::make_pair("K",interpK));
    auto interpKU = std::make_shared<HDBinterpolator>();
    m_interpIRF.insert(std::make_pair("KU",interpKU));
    auto interpKUXDerivative = std::make_shared<HDBinterpolator>();
    m_interpIRF.insert(std::make_pair("KUXDerivative",interpKUXDerivative));
    auto interpKU2 = std::make_shared<HDBinterpolator>();
    m_interpIRF.insert(std::make_pair("KU2",interpKU2));
    AllocateAll();
  }

  //
  // Setters
  //

  void Body::SetHorizontalPositionInWorld(const mathutils::Vector3d<double> &horizontal_position_in_world_frame) {
    m_horizontal_position_in_world_frame = horizontal_position_in_world_frame;
  }

  void Body::SetComputationPointInBodyFrame(const mathutils::Vector3d<double> &computation_point_in_body_frame) {
    m_computation_point_in_body_frame = computation_point_in_body_frame;
  }

  void Body::SetWaveReferencePointInBodyFrame(const mathutils::Vector2d<double> &wave_reference_point_in_body_frame) {
    m_wave_reference_point_in_body_frame = wave_reference_point_in_body_frame;
  }

  void Body::SetForceMask(const mathutils::Vector6d<bool> &mask) {
    m_forceMask.SetMask(mask);
  }

  void Body::SetDiffraction(unsigned int iangle, const Eigen::MatrixXcd &diffractionMatrix) {
    assert(iangle < m_HDB->GetWaveDirectionDiscretization().size());
    assert(diffractionMatrix.rows() == 6);
    assert(diffractionMatrix.cols() == m_HDB->GetFrequencyDiscretization().size());
    m_diffraction[iangle] = diffractionMatrix;
  }

  void Body::SetDiffraction(unsigned int iangle, unsigned int iw, const Eigen::VectorXcd &diffractionVector){
    assert(iangle < m_HDB->GetWaveDirectionDiscretization().size());
    assert(iw < m_HDB->GetFrequencyDiscretization().size());
    assert(diffractionVector.rows() == 6);
    assert(diffractionVector.cols() == 1);
    m_diffraction[iangle].col(iw) = diffractionVector;
  }

  void Body::SetXDerivativeDiffraction(unsigned int iangle, const Eigen::MatrixXcd &diffractionXDerivativeMatrix) {
    assert(iangle < m_HDB->GetWaveDirectionDiscretization().size());
    assert(diffractionXDerivativeMatrix.rows() == 6);
    assert(diffractionXDerivativeMatrix.cols() == m_HDB->GetFrequencyDiscretization().size());
    m_diffraction_x_derivative[iangle] = diffractionXDerivativeMatrix;
  }

  void Body::SetXDerivativeDiffraction(unsigned int iangle, unsigned int iw, const Eigen::VectorXcd &diffractionXDerivativeVector){
    assert(iangle < m_HDB->GetWaveDirectionDiscretization().size());
    assert(iw < m_HDB->GetFrequencyDiscretization().size());
    assert(diffractionXDerivativeVector.rows() == 6);
    assert(diffractionXDerivativeVector.cols() == 1);
    m_diffraction_x_derivative[iangle].col(iw) = diffractionXDerivativeVector;
  }

  void Body::SetFroudeKrylov(unsigned int iangle, const Eigen::MatrixXcd &froudeKrylovMatrix) {
    assert(iangle < m_HDB->GetWaveDirectionDiscretization().size());
    assert(froudeKrylovMatrix.rows() == 6);
    assert(froudeKrylovMatrix.cols() == m_HDB->GetFrequencyDiscretization().size());
    m_froudeKrylov[iangle] = froudeKrylovMatrix;
  }

  void Body::SetFroudeKrylov(unsigned int iangle, unsigned int iw, const Eigen::VectorXcd &froudeKrylovVector) {
    assert(iangle < m_HDB->GetWaveDirectionDiscretization().size());
    assert(iw < m_HDB->GetFrequencyDiscretization().size());
    assert(froudeKrylovVector.rows() == 6);
    assert(froudeKrylovVector.cols() == 1);
    m_froudeKrylov[iangle].col(iw) = froudeKrylovVector;
  }

  void Body::SetXDerivativeFroudeKrylov(unsigned int iangle, const Eigen::MatrixXcd &froudeKrylovXDerivativeMatrix) {
    assert(iangle < m_HDB->GetWaveDirectionDiscretization().size());
    assert(froudeKrylovXDerivativeMatrix.rows() == 6);
    assert(froudeKrylovXDerivativeMatrix.cols() == m_HDB->GetFrequencyDiscretization().size());
    m_froudeKrylov_x_derivative[iangle] = froudeKrylovXDerivativeMatrix;
  }

  void Body::SetXDerivativeFroudeKrylov(unsigned int iangle, unsigned int iw, const Eigen::VectorXcd &froudeKrylovXDerivativeVector) {
    assert(iangle < m_HDB->GetWaveDirectionDiscretization().size());
    assert(iw < m_HDB->GetFrequencyDiscretization().size());
    assert(froudeKrylovXDerivativeVector.rows() == 6);
    assert(froudeKrylovXDerivativeVector.cols() == 1);
    m_froudeKrylov_x_derivative[iangle].col(iw) = froudeKrylovXDerivativeVector;
  }

  void Body::SetExcitation(unsigned int iangle, const Eigen::MatrixXcd &excitationMatrix) {
    assert(iangle < m_HDB->GetWaveDirectionDiscretization().size());
    assert(excitationMatrix.rows() == 6);
    assert(excitationMatrix.cols() == m_HDB->GetFrequencyDiscretization().size());
    m_excitation[iangle] = excitationMatrix;
  }

  void Body::SetExcitation(unsigned int iangle, unsigned int iw, const Eigen::VectorXcd &excitationVector) {
    assert(iangle < m_HDB->GetWaveDirectionDiscretization().size());
    assert(iw < m_HDB->GetFrequencyDiscretization().size());
    assert(excitationVector.rows() == 6);
    assert(excitationVector.cols() == 1);
    m_excitation[iangle].col(iw) = excitationVector;
  }

  void Body::SetXDerivativeExcitation(unsigned int iangle, const Eigen::MatrixXcd &excitationXDerivativeMatrix) {
    assert(iangle < m_HDB->GetWaveDirectionDiscretization().size());
    assert(excitationXDerivativeMatrix.rows() == 6);
    assert(excitationXDerivativeMatrix.cols() == m_HDB->GetFrequencyDiscretization().size());
    m_excitation_x_derivative[iangle] = excitationXDerivativeMatrix;
  }

  void Body::SetXDerivativeExcitation(unsigned int iangle, unsigned int iw, const Eigen::VectorXcd &excitationXDerivativeVector) {
    assert(iangle < m_HDB->GetWaveDirectionDiscretization().size());
    assert(iw < m_HDB->GetFrequencyDiscretization().size());
    assert(excitationXDerivativeVector.rows() == 6);
    assert(excitationXDerivativeVector.cols() == 1);
    m_excitation_x_derivative[iangle].col(iw) = excitationXDerivativeVector;
  }

  void Body::ComputeExcitation() {

    // This method computes the excitation loads from the diffraction loads and the Froude-Krylov loads.

    assert(m_diffraction.size() == m_froudeKrylov.size());
    for (unsigned int iangle = 0; iangle < m_diffraction.size(); ++iangle) {
      m_excitation[iangle] = m_diffraction[iangle] + m_froudeKrylov[iangle];
    }
  }

  void Body::ComputeXDerivativeExcitation() {

    // This method computes the x-derivative of the excitation loads from x-derivative of both the diffraction loads and the Froude-Krylov loads.

    assert(m_diffraction_x_derivative.size() == m_froudeKrylov_x_derivative.size());
    for (unsigned int iangle = 0; iangle < m_diffraction_x_derivative.size(); ++iangle) {
      m_excitation_x_derivative[iangle] = m_diffraction_x_derivative[iangle] + m_froudeKrylov_x_derivative[iangle];
    }
  }

  void Body::SetInfiniteAddedMass(Body *BodyMotion, const Matrix66 &CMInf) {
    m_infiniteAddedMass[BodyMotion] = CMInf;
  }

  void Body::SetXDerivativeInfiniteAddedMass(Body *BodyMotion, const Matrix66 &XDerivativeCMInf) {
    m_infiniteAddedMass_x_derivative[BodyMotion] = XDerivativeCMInf;
  }

  void Body::SetZeroFreqAddedMass(Body *BodyMotion, const Matrix66 &CMZero) {
    m_zeroFreqAddedMass[BodyMotion] = CMZero;
  }

  void Body::SetXDerivativeZeroFreqAddedMass(Body *BodyMotion, const Matrix66 &XDerivativeCMZero) {
    m_zeroFreqAddedMass_x_derivative[BodyMotion] = XDerivativeCMZero;
  }

//  void Body::SetRadiationMask(hdb5_io::Body *BodyMotion, const Matrix66b &mask){
//    m_radiationMask[BodyMotion] = mask;
//  }

  void Body::SetRadiationMask(hdb5_io::Body *BodyMotion, const Matrix66b &mask) {
    assert(mask.maxCoeff() <= 1 && mask.minCoeff() >= 0);
    m_radiationMask[BodyMotion] = mask;
  }

  void Body::SetRAO(unsigned int iangle, const Eigen::MatrixXcd &RAO) {
    assert(iangle < m_HDB->GetWaveDirectionDiscretization().size());
    assert(RAO.rows() == 6);
    assert(RAO.cols() == m_HDB->GetFrequencyDiscretization().size());
    m_RAO[iangle] = RAO;
    m_isRAO = true;
  }

  void Body::SetRAO(unsigned int iangle, unsigned int iw, const Eigen::VectorXcd &RAO) {
    assert(iangle < m_HDB->GetWaveDirectionDiscretization().size());
    assert(iw < m_HDB->GetFrequencyDiscretization().size());
    assert(RAO.rows() == 6);
    assert(RAO.cols() == 1);
    m_RAO[iangle].col(iw) = RAO;
    m_isRAO = true;
  }

  void Body::SetAddedMass(Body *BodyMotion, const std::vector<Matrix66> &listData) {
    assert(listData.size() == m_HDB->GetFrequencyDiscretization().size());
    m_addedMass.insert(std::make_pair(BodyMotion,listData));
  }

  void Body::SetXDerivativeAddedMass(Body *BodyMotion, const std::vector<Matrix66> &listData) {
    assert(listData.size() == m_HDB->GetFrequencyDiscretization().size());
    m_addedMass_x_derivative.insert(std::make_pair(BodyMotion,listData));
  }

  void Body::SetRadiationDamping(Body *BodyMotion, const std::vector<Matrix66> &listData) {
    assert(listData.size() == m_HDB->GetFrequencyDiscretization().size());
    m_radiationDamping.insert(std::make_pair(BodyMotion,listData));
  }

  void Body::SetXDerivativeRadiationDamping(Body *BodyMotion, const std::vector<Matrix66> &listData) {
    assert(listData.size() == m_HDB->GetFrequencyDiscretization().size());
    m_radiationDamping_x_damping.insert(std::make_pair(BodyMotion,listData));
  }

  void Body::AddAddedMass(Body *BodyMotion, const Matrix66 &Data) {
    if (m_addedMass.count(BodyMotion) == 0) {
      std::vector<Matrix66> tempMat;
      tempMat.reserve(m_HDB->GetFrequencyDiscretization().size());
      m_addedMass.insert(std::make_pair(BodyMotion, tempMat));
    }
    m_addedMass.at(BodyMotion).push_back(Data);
  }

  void Body::AddXDerivativeAddedMass(Body *BodyMotion, const Matrix66 &Data) {
    if (m_addedMass_x_derivative.count(BodyMotion) == 0) {
      std::vector<Matrix66> tempMat;
      tempMat.reserve(m_HDB->GetFrequencyDiscretization().size());
      m_addedMass_x_derivative.insert(std::make_pair(BodyMotion, tempMat));
    }
    m_addedMass_x_derivative.at(BodyMotion).push_back(Data);
  }

  void Body::AddRadiationDamping(Body *BodyMotion, const Matrix66 &Data) {
    if (m_radiationDamping.count(BodyMotion) == 0) {
      std::vector<Matrix66> tempMat;
      tempMat.reserve(m_HDB->GetFrequencyDiscretization().size());
      m_radiationDamping.insert(std::make_pair(BodyMotion, tempMat));
    }
    m_radiationDamping.at(BodyMotion).push_back(Data);
  }

  void Body::AddXDerivativeRadiationDamping(Body *BodyMotion, const Matrix66 &Data) {
    if (m_radiationDamping_x_damping.count(BodyMotion) == 0) {
      std::vector<Matrix66> tempMat;
      tempMat.reserve(m_HDB->GetFrequencyDiscretization().size());
      m_radiationDamping_x_damping.insert(std::make_pair(BodyMotion, tempMat));
    }
    m_radiationDamping_x_damping.at(BodyMotion).push_back(Data);
  }

  void Body::SetIRF(Body *BodyMotion, const std::string &IRF_type, const std::vector<Eigen::MatrixXd> &listData) {

    unsigned int idof = 0;

    if (m_interpIRF.at(IRF_type)->count(idof) == 0) {

      for (auto &data: listData) {
        assert(data.rows() == 6);

        auto table = std::make_shared<mathutils::LookupTable1D<double, mathutils::Vector6d<double>>>(mathutils::LINEAR);

        auto EigenTime = m_HDB->GetTimeDiscretization();
        assert(data.cols() == EigenTime.size());
        std::vector<double> time(&EigenTime(0, 0), EigenTime.data() + EigenTime.size());
        table->SetX(time);

        std::vector<mathutils::Vector6d<double>> vdata;
        for (unsigned int j = 0; j < data.cols(); ++j) {
          vdata.emplace_back(data.col(j));
        }
        table->AddY(BodyMotion->GetName(), vdata);

        m_interpIRF.at(IRF_type)->insert(std::make_pair(idof, table));

        idof++;
      }
    } else {
      for (auto &data: listData) {
        assert(data.rows() == 6);
        std::vector<mathutils::Vector6d<double>> vdata;
        for (unsigned int j = 0; j < data.cols(); ++j) {
          vdata.emplace_back(data.col(j));
        }
        m_interpIRF.at(IRF_type)->at(idof)->AddY(BodyMotion->GetName(), vdata);
        idof++;
      }
    }

    m_isIRF = true;

  }

  void Body::SetStiffnessMatrix(const Matrix33 &hydrostaticStiffnessMatrix) {
    m_hydrostaticStiffnessMatrix = hydrostaticStiffnessMatrix;
    m_isHydrostatic = true;
  }

  void Body::SetStiffnessMatrix(const Matrix66 &hydrostaticStiffnessMatrix) {
    m_hydrostaticStiffnessMatrix = hydrostaticStiffnessMatrix.block<3, 3>(2, 2);
    m_isHydrostatic = true;
  }

  void Body::SetInertia(const Matrix66 &inertiaMatrix) {
    m_inertia = inertiaMatrix;
    m_isInertia = true;
  }

  void Body::SetMooring(const Matrix66 &mooringMatrix) {
    m_mooringStiffnessMatrix = mooringMatrix;
    m_isMooring = true;
  }

  void Body::SetLinearDamping(const Matrix66 &linearDampingMatrix) {
    m_linearDampingMatrix = linearDampingMatrix;
    m_isDamping = true;
  }

  void Body::SetWaveDrift(const std::shared_ptr<WaveDrift> &wavedrift) {
    m_waveDrift = wavedrift;
    m_isWaveDrift = true;
  }

#ifdef MESH_SUPPORT
  void
  Body::LoadMesh(const std::vector<mathutils::Vector3d<double>> &vertices, const std::vector<Eigen::VectorXi> &faces) {
    m_mesh->Load(vertices, faces);
  }
#endif

  void Body::AddModalCoefficients(Body *BodyMotion, const std::vector<PoleResidue> &modalCoefficients) {
    if (m_modalCoefficients.count(BodyMotion)>0) {
      auto coeff = m_modalCoefficients.find(BodyMotion);
      coeff->second.emplace_back(modalCoefficients);
    }else {
      std::vector<std::vector<PoleResidue>> coefficients;
      coefficients.emplace_back(modalCoefficients);
      m_modalCoefficients[BodyMotion] = coefficients;
//      auto paired = std::make_pair(BodyMotion, coefficients);
//      m_modalCoefficients.insert(paired);
    }

  }

  void Body::ClearModalCoefficients(Body *BodyMotion) {

    // This method clears the modal coefficients of a body.

    m_modalCoefficients[BodyMotion].clear();

  }

  //
  // Getters
  //

  bool Body::HasRAO() const {
    return m_isRAO;
  }

  bool Body::HasModal(Body *BodyMotion) const {
    return m_modalCoefficients.count(BodyMotion) > 0;
  }

  bool Body::HasIRF() const {
    return m_isIRF;
  }

  bool Body::HasInertia() const {
    return m_isInertia;
  }

  bool Body::HasHydrostatic() const {
    return m_isHydrostatic;
  }

  bool Body::HasMooring() const {
    return m_isMooring;
  }

  bool Body::HasDamping() const {
    return m_isDamping;
  }

  bool Body::HasZeroFreqAddedMass(Body *BodyMotion) const {
    return m_zeroFreqAddedMass.count(BodyMotion) > 0;
  }

  bool Body::HasWaveDrift() const {
    return m_isWaveDrift;
  }

  mathutils::Vector3d<double> Body::GetHorizontalPositionInWorld() const {
    return m_horizontal_position_in_world_frame;
  }

  mathutils::Vector3d<double> Body::GetComputationPointInBodyFrame() const {
    return m_computation_point_in_body_frame;
  }

  mathutils::Vector2d<double> Body::GetWaveReferencePointInBodyFrame() const {
    return m_wave_reference_point_in_body_frame;
  }

  Mask Body::GetForceMask() const {
    return m_forceMask;
  }

#ifdef MESH_SUPPORT
  Mesh *Body::GetMesh() const {
    return m_mesh.get();
  }
#endif

  Matrix33 Body::GetHydrostaticStiffnessMatrix() const {
    return m_hydrostaticStiffnessMatrix;
  }

  Matrix66 Body::GetInertiaMatrix() const {
    return m_inertia;
  }

  Matrix66 Body::GetMooringMatrix() const {
    return m_mooringStiffnessMatrix;
  }

  Matrix66 Body::GetDampingMatrix() const {
    return m_linearDampingMatrix;
  }

  Eigen::MatrixXcd Body::GetDiffraction(const unsigned int iangle) const {
//    assert(iangle < this->GetNbWaveDirections());
    return m_diffraction[iangle];
  }

  Eigen::VectorXcd Body::GetDiffraction(const unsigned int iangle, const unsigned iforce) const {
//    assert(iangle < this->GetNbWaveDirections());
//    assert(iforce < 6);
    return m_diffraction[iangle].row(iforce);
  }

  Eigen::MatrixXcd Body::GetXDerivativeDiffraction(const unsigned int iangle) const {
    return m_diffraction_x_derivative[iangle];
  }

  Eigen::VectorXcd Body::GetXDerivativeDiffraction(const unsigned int iangle, const unsigned iforce) const {
    return m_diffraction_x_derivative[iangle].row(iforce);
  }

  Eigen::MatrixXcd Body::GetFroudeKrylov(const unsigned int iangle) const {
//    assert(iangle < this->GetNbWaveDirections());
    return m_froudeKrylov[iangle];
  }

  Eigen::VectorXcd Body::GetFroudeKrylov(const unsigned int iangle, const unsigned iforce) const {
//    assert(iangle < this->GetNbWaveDirections());
//    assert(iforce < 6);
    return m_froudeKrylov[iangle].row(iforce);
  }

  Eigen::MatrixXcd Body::GetXDerivativeFroudeKrylov(const unsigned int iangle) const {
    return m_froudeKrylov_x_derivative[iangle];
  }

  Eigen::VectorXcd Body::GetXDerivativeFroudeKrylov(const unsigned int iangle, const unsigned iforce) const {
    return m_froudeKrylov_x_derivative[iangle].row(iforce);
  }

  Eigen::MatrixXcd Body::GetExcitation(const unsigned int iangle) const {
//    assert(iangle < this->GetNbWaveDirections());
    return m_excitation[iangle];
  }

  Eigen::VectorXcd Body::GetExcitation(const unsigned int iangle, const unsigned iforce) const {
//    assert(iangle < this->GetNbWaveDirections());
//    assert(iforce < 6);
    return m_excitation[iangle].row(iforce);
  }

  Eigen::MatrixXcd Body::GetXDerivativeExcitation(const unsigned int iangle) const {
    return m_excitation_x_derivative[iangle];
  }

  Eigen::VectorXcd Body::GetXDerivativeExcitation(const unsigned int iangle, const unsigned iforce) const {
    return m_excitation_x_derivative[iangle].row(iforce);
  }

  Matrix66 Body::GetInfiniteAddedMass(Body *BodyMotion) const {
    return m_infiniteAddedMass.at(BodyMotion);
  }

  Matrix66 Body::GetXDerivativeInfiniteAddedMass(Body *BodyMotion) const {
    return m_infiniteAddedMass_x_derivative.at(BodyMotion);
  }

  Matrix66 Body::GetZeroFreqAddedMass(hdb5_io::Body *BodyMotion) const {
    return m_zeroFreqAddedMass.at(BodyMotion);
  }

  Matrix66 Body::GetXDerivativeZeroFreqAddedMass(hdb5_io::Body *BodyMotion) const {
    return m_zeroFreqAddedMass_x_derivative.at(BodyMotion);
  }

  Matrix66b Body::GetRadiationMask(Body *BodyMotion) const {
    return m_radiationMask.at(BodyMotion);
  }

  Matrix66 Body::GetSelfInfiniteAddedMass() {
    return m_infiniteAddedMass[this];
  }

  Matrix66 Body::GetSelfXDerivativeInfiniteAddedMass() {
    return m_infiniteAddedMass_x_derivative[this];
  }

  Matrix66 Body::GetSelfZeroAddedMass() {
    return m_zeroFreqAddedMass[this];
  }

  Matrix66 Body::GetSelfXDerivativeZeroAddedMass() {
    return m_zeroFreqAddedMass_x_derivative[this];
  }

  Eigen::MatrixXcd Body::GetRAO(const unsigned int iangle) const {
//    assert(iangle < this->GetNbWaveDirections());
    return m_RAO[iangle];
  }

  Eigen::VectorXcd Body::GetRAO(const unsigned int iangle, const unsigned iforce) const {
//    assert(iangle < this->GetNbWaveDirections());
    assert(iforce < 6);
    return m_RAO[iangle].row(iforce);
  }

  Eigen::MatrixXd
  Body::GetIRFInterpolatedData(Body *BodyMotion, const std::string& IRF_type, unsigned int idof,
                               mathutils::VectorN<double> frequencies) {

    Eigen::MatrixXd data(6, frequencies.size());
    for (unsigned int i = 0; i < frequencies.size(); i++) {
      data.col(i) = m_interpIRF.at(IRF_type)->at(idof)->Eval(BodyMotion->GetName(), frequencies(i));
    }
    return data;

  }

  std::vector<PoleResidue> Body::GetModalCoefficients(Body *BodyMotion, int idof) {
    return m_modalCoefficients.at(BodyMotion)[idof];
  }

  PoleResidue Body::GetModalCoefficients(Body *BodyMotion, int idof, int iforce) {
    return m_modalCoefficients.at(BodyMotion)[idof][iforce];
  }


  WaveDrift* Body::GetWaveDrift() const {
    return m_waveDrift.get();
  }

  void Body::AllocateAll() {

    // This subroutine allocates the arrays for the hdb.

    auto nDirections = m_HDB->GetWaveDirectionDiscretization().size();
    auto nFrequencies = m_HDB->GetFrequencyDiscretization().size();
    m_diffraction.reserve((unsigned long) nDirections);
    m_diffraction_x_derivative.reserve((unsigned long) nDirections);
    m_froudeKrylov.reserve((unsigned long) nDirections);
    m_froudeKrylov_x_derivative.reserve((unsigned long) nDirections);
    m_excitation.reserve((unsigned long) nDirections);
    m_excitation_x_derivative.reserve((unsigned long) nDirections);
    m_RAO.reserve((unsigned long) nDirections);

    for (int i = 0; i < nDirections; ++i) {
      Eigen::MatrixXcd mat(6, nFrequencies);
      m_diffraction.push_back(mat);
      m_diffraction_x_derivative.push_back(mat);
      m_froudeKrylov.push_back(mat);
      m_froudeKrylov_x_derivative.push_back(mat);
      m_excitation.push_back(mat);
      m_excitation_x_derivative.push_back(mat);
      m_RAO.push_back(mat);
    }

    /// --> Allocating arrays for radiation
    m_radiationMask.reserve(m_HDB->GetNbBodies());

  }

  Body::HDBinterpolator *Body::GetIRFInterpolator(const std::string& IRF_type) const {
    return m_interpIRF.at(IRF_type).get();
  }

  Eigen::MatrixXd Body::GetAddedMass(Body *BodyMotion, unsigned int imotion) const {
    Eigen::MatrixXd AM = Eigen::MatrixXd::Zero(6, m_HDB->GetFrequencyDiscretization().size());

    for (int iforce = 0; iforce < 6; ++iforce) {
      for (int iw = 0; iw < m_HDB->GetFrequencyDiscretization().size(); ++iw) {
        AM(iforce, iw) = m_addedMass.at(BodyMotion)[iw](iforce, imotion);
      }
    }
    return AM;
  }

  Eigen::MatrixXd Body::GetXDerivativeAddedMass(Body *BodyMotion, unsigned int imotion) const {
    Eigen::MatrixXd AM = Eigen::MatrixXd::Zero(6, m_HDB->GetFrequencyDiscretization().size());

    for (int iforce = 0; iforce < 6; ++iforce) {
      for (int iw = 0; iw < m_HDB->GetFrequencyDiscretization().size(); ++iw) {
        AM(iforce, iw) = m_addedMass_x_derivative.at(BodyMotion)[iw](iforce, imotion);
      }
    }
    return AM;
  }

  Eigen::MatrixXd Body::GetRadiationDamping(Body *BodyMotion, unsigned int imotion) const {
    Eigen::MatrixXd RD = Eigen::MatrixXd::Zero(6, m_HDB->GetFrequencyDiscretization().size());

    for (int iforce = 0; iforce < 6; ++iforce) {
      for (int iw = 0; iw < m_HDB->GetFrequencyDiscretization().size(); ++iw) {
        RD(iforce, iw) = m_radiationDamping.at(BodyMotion)[iw](iforce, imotion);
      }
    }
    return RD;
  }

  Eigen::MatrixXd Body::GetXDerivativeRadiationDamping(Body *BodyMotion, unsigned int imotion) const {
    Eigen::MatrixXd RD = Eigen::MatrixXd::Zero(6, m_HDB->GetFrequencyDiscretization().size());

    for (int iforce = 0; iforce < 6; ++iforce) {
      for (int iw = 0; iw < m_HDB->GetFrequencyDiscretization().size(); ++iw) {
        RD(iforce, iw) = m_radiationDamping_x_damping.at(BodyMotion)[iw](iforce, imotion);
      }
    }
    return RD;
  }

  Matrix66 Body::GetAddedMassPerFrequency(Body *BodyMotion, unsigned int iomega) const {
    return m_addedMass.at(BodyMotion)[iomega];
  }

  Matrix66 Body::GetXDerivativeAddedMassPerFrequency(Body *BodyMotion, unsigned int iomega) const {
    return m_addedMass_x_derivative.at(BodyMotion)[iomega];
  }

  Matrix66 Body::GetRadiationDampingPerFrequency(Body *BodyMotion, unsigned int iomega) const {
    return m_radiationDamping.at(BodyMotion)[iomega];
  }

  Matrix66 Body::GetXDerivativeRadiationDampingPerFrequency(Body *BodyMotion, unsigned int iomega) const {
    return m_radiationDamping_x_damping.at(BodyMotion)[iomega];
  }

#ifdef USE_VTK

  void Body::VisualizeMesh() const {
    // Creation of a VTK mesh.
    meshoui::VTKMesh vtkmesh = meshoui::VTKMesh(*m_mesh);

    // Visualization.
    vtkmesh.Visualize();
  }

#endif

}
