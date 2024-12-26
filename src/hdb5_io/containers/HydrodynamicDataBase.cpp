//
// Created by lletourn on 26/02/20.
//

#include "HydrodynamicDataBase.h"

#include "WaveDrift.h"

#include "Body.h"

namespace hdb5_io {

  void HydrodynamicDataBase::SetCreationDate(std::string date) {
    m_creationDate = date;
  }

  std::string HydrodynamicDataBase::GetCreationDate() const {
    return m_creationDate;
  }

  void HydrodynamicDataBase::SetSolver(std::string solver) {
    m_solver = solver;
  }

  std::string HydrodynamicDataBase::GetSolver() const {
    return m_solver;
  }

  void HydrodynamicDataBase::SetNormalizedVersionString(std::string commit_hash) {
    m_is_commit_hash = true;
    m_commit_hash = commit_hash;
  }

  std::string HydrodynamicDataBase::GetNormalizedVersionString() const {
    return m_commit_hash;
  }

  bool HydrodynamicDataBase::IsNormalizedVersionString() const {
    return m_is_commit_hash;
  }

  void HydrodynamicDataBase::SetVersion(double version) {
    m_version = version;
  }

  double HydrodynamicDataBase::GetVersion() const {
    return m_version;
  }

  void HydrodynamicDataBase::SetGravityAcceleration(double g) {
    m_gravityAcceleration = g;
  }

  double HydrodynamicDataBase::GetGravityAcceleration() const {
    return m_gravityAcceleration;
  }

  void HydrodynamicDataBase::SetWaterDensity(double rho) {
    m_waterDensity = rho;
  }

  double HydrodynamicDataBase::GetWaterDensity() const {
    return m_waterDensity;
  }

  void HydrodynamicDataBase::SetWaterDepth(double h) {
    m_waterDepth = h;
  }

  double HydrodynamicDataBase::GetWaterDepth() const {
    return m_waterDepth;
  }

  void HydrodynamicDataBase::SetNormalizationLength(double L) {
    m_normalizationLength = L;
  }

  double HydrodynamicDataBase::GetNormalizationLength() const {
    return m_normalizationLength;
  }

  Body *HydrodynamicDataBase::NewBody(unsigned int id, const std::string &name) {
    auto bodyName = name;
    if (bodyName.empty()) {
      std::cerr<<"empty body name : " << id << std::endl;
      std::exit(EXIT_FAILURE);
    }
    for (auto & body : m_bodies) {
      if (bodyName == body->GetName()) {
        std::cerr << "body " << id << " and body " << body->GetID() << " have the same name !" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    m_bodies.push_back(std::make_shared<Body>(id, bodyName, this));
    return (m_bodies.back()).get();
  }

  Body *HydrodynamicDataBase::GetBody(int id) const {
    return m_bodies[id].get();
  }

  Body *HydrodynamicDataBase::GetBody(const std::string& name) const {
    for (auto &body : m_bodies) {
      if (body->GetName() == name) {
        return body.get();
      };
    }
    std::cerr << "Body " << name << " not found in HDB. Abort." << std::endl;
    exit(1);
    return nullptr;
  }

  void HydrodynamicDataBase::SetNbBodies(int nb) {
    m_nbody = nb;
  }

  int HydrodynamicDataBase::GetNbBodies() const {
    return m_nbody;
  }

  void HydrodynamicDataBase::SetFrequencyDiscretization(const mathutils::VectorN<double> &frequency) {
    m_frequencyDiscretization = frequency;
  }

  void HydrodynamicDataBase::SetWaveDirectionDiscretization(const mathutils::VectorN<double> &directions) {
    m_waveDirectionDiscretization = directions;
  }

  void HydrodynamicDataBase::SetTimeDiscretization(const mathutils::VectorN<double> &time) {
    m_timeDiscretization = time;
  }

  mathutils::VectorN<double> HydrodynamicDataBase::GetFrequencyDiscretization() const {
    return m_frequencyDiscretization;
  }

  mathutils::VectorN<double> HydrodynamicDataBase::GetWaveDirectionDiscretization() const {
    return m_waveDirectionDiscretization;
  }

  mathutils::VectorN<double> HydrodynamicDataBase::GetTimeDiscretization() const {
    return m_timeDiscretization;
  }

  double HydrodynamicDataBase::GetMinFrequency() const {
    return m_frequencyDiscretization[0];
  }

  double HydrodynamicDataBase::GetMaxFrequency() const {
    return m_frequencyDiscretization[m_frequencyDiscretization.size()-1];
  }

  void HydrodynamicDataBase::SetKochin(const std::shared_ptr<Kochin> &kochin) {
    m_kochin = kochin;
  }

  Kochin* HydrodynamicDataBase::GetKochin() const {
    return m_kochin.get();
  }

  void HydrodynamicDataBase::SetVF() {
    m_isVF = true;
  }

  bool HydrodynamicDataBase::GetVF() const {
    return m_isVF;
  }

  void HydrodynamicDataBase::SetVFRelaxed(const int &relaxed) {
    m_VF_relaxed = relaxed;
  }

  int HydrodynamicDataBase::GetVFRelaxed() const {
    return m_VF_relaxed;
  }

  void HydrodynamicDataBase::SetVFMaxOrder(const int &order) {
    m_VF_max_order = order;
  }

  int HydrodynamicDataBase::GetVFMaxOrder() const {
    return m_VF_max_order;
  }

  void HydrodynamicDataBase::SetVFTolerance(const double &tolerance) {
    m_VF_tolerance = tolerance;
  }

  double HydrodynamicDataBase::GetVFTolerance() const {
    return m_VF_tolerance;
  }

  void HydrodynamicDataBase::SetWaveField() {
    m_isWaveField = true;
  }

  bool HydrodynamicDataBase::GetWaveField() const {
    return m_isWaveField;
  }

  void HydrodynamicDataBase::SetSymmetries() {
    m_isSymmetries = true;
  }

  bool HydrodynamicDataBase::GetSymmetries() const {
    return m_isSymmetries;
  }

  void HydrodynamicDataBase::SetSymBottom(const bool &sym_bottom) {
    m_sym_bottom = sym_bottom;
  }

  bool HydrodynamicDataBase::GetSymBottom() const {
    return m_sym_bottom;
  }

  void HydrodynamicDataBase::SetSymXOZ(const bool &sym_xoz) {
    m_sym_xoz = sym_xoz;
  }

  bool HydrodynamicDataBase::GetSymXOZ() const {
    return m_sym_xoz;
  }

  void HydrodynamicDataBase::SetSymYOZ(const bool &sym_yoz) {
    m_sym_yoz = sym_yoz;
  }

  bool HydrodynamicDataBase::GetSymYOZ() const {
    return m_sym_yoz;
  }

  void HydrodynamicDataBase::SetExpertParameters() {
    m_isExpertParameters = true;
  }

  bool HydrodynamicDataBase::GetExpertParameters() const {
    return m_isExpertParameters;
  }

  void HydrodynamicDataBase::SetSurfaceIntegrationOrder(const int &order) {
    m_surface_integration_order = order;
  }

  int HydrodynamicDataBase::GetSurfaceIntegrationOrder() const {
    return m_surface_integration_order;
  }

  void HydrodynamicDataBase::SetGreenFunction(const std::string &green_function) {
    m_Green_function = green_function;
  }

  std::string HydrodynamicDataBase::GetGreenFunction() const {
    return m_Green_function;
  }

  void HydrodynamicDataBase::SetCrmax(const int &crmax) {
    m_Crmax = crmax;
  }

  int HydrodynamicDataBase::GetCrmax() const {
    return m_Crmax;
  }

  void HydrodynamicDataBase::IsXDerivative() {
    m_is_x_derivative = true;
  }

  bool HydrodynamicDataBase::GetIsXDerivative() const {
    return m_is_x_derivative;
  }

} // namespace hdb5_io
