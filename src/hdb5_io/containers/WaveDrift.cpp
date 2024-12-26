//
// Created by lletourn on 07/04/20.
//

#include "WaveDrift.h"

namespace hdb5_io {


  WaveDrift::WaveDrift() : m_symmetry_X(false), m_symmetry_Y(false) {
    m_data = std::make_unique<mathutils::LookupTable2d<double>>();
  }

  void WaveDrift::SetSymmetries(bool symmetry_X, bool symmetry_Y) {
    m_symmetry_X = symmetry_X;
    m_symmetry_Y = symmetry_Y;
  }

  std::vector<bool> WaveDrift::GetSymmetries() const {
    return {m_symmetry_X, m_symmetry_Y};
  }

  void WaveDrift::SetFrequencies(const mathutils::VectorN<double> &frequencies) {
    std::vector<double> freq(&frequencies(0, 0), frequencies.data() + frequencies.size());
    m_data->SetX(freq);
  }

  void WaveDrift::SetWaveDirections(const mathutils::VectorN<double> &angles) {
    std::vector<double> ang(&angles(0, 0), angles.data() + angles.size());
    m_data->SetY(ang);
  }

  void WaveDrift::AddData(const std::string &name, const std::vector<double> &coeffs) {
    m_data->AddData(name, coeffs);
  }

  double WaveDrift::Eval(const std::string &name, double frequency, double angle) const {
    return m_data->Eval(name, frequency, angle);
  }

  void WaveDrift::SetKochinStep(const double &kochin_step) {
    m_kochin_step = kochin_step;
  }

  double WaveDrift::GetKochinStep() const {
    return m_kochin_step;
  }

}