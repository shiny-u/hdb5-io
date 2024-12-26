//
// Created by pierre-yves on 13/01/2021.
//

#include "Kochin.h"
#include "HydrodynamicDataBase.h"

namespace hdb5_io {

  Kochin::Kochin(HydrodynamicDataBase *hdb, const double &kochin_step, int NbDir) : m_HDB(hdb), m_kochin_step(kochin_step){

    // Constructor of the class.

    // Number of angular steps.
    ComputeNbKochinAngles();

    // Allocations.
    auto nFrequencies = m_HDB->GetFrequencyDiscretization().size();
    m_kochin_diffraction.reserve((unsigned long) NbDir);
    m_kochin_diffraction_derivate.reserve((unsigned long) NbDir);

    for (int i = 0; i < NbDir; ++i) {
      Eigen::MatrixXcd mat(m_nb_kochin_angle, nFrequencies);
      m_kochin_diffraction.push_back(mat);
      m_kochin_diffraction_derivate.push_back(mat);
    }

  }

  double Kochin::GetKochinStep() const {
    return m_kochin_step;
  }

  void Kochin::ComputeNbKochinAngles(){

    // Compute the number of Kochin angles.

    m_nb_kochin_angle = 0;
    for (double theta = 0; theta <= MU_2PI; theta += m_kochin_step) {
      ++m_nb_kochin_angle;
    }

  }

  int Kochin::GetNbKochinAngles() const {

    // Getter of the number of Kochin angles.

    return m_nb_kochin_angle;

  }

  // Setter for the number of Kochin wave directions.
  void Kochin::SetWaveDirectionKochin(const mathutils::VectorN<double> &directions) {

    m_waveDirectionKochin = directions;

  }

  mathutils::VectorN<double> Kochin::GetWaveDirectionKochin() const {

    // Getter for the Kochin wave direction vector.

    return m_waveDirectionKochin;

  }

  void Kochin::SetDiffractionKochin(unsigned int iwave, unsigned int iw, const Eigen::VectorXcd &diffractionKochinVector) {

    // Setter of the diffraction Kochin function for a single wave frequency, all angles and a single wave direction.

    assert(iwave < m_HDB->GetWaveDirectionDiscretization().size());
    assert(iw < m_HDB->GetFrequencyDiscretization().size());
    assert(diffractionKochinVector.rows() == m_nb_kochin_angle);
    assert(diffractionKochinVector.cols() == 1);
    m_kochin_diffraction[iwave].col(iw) = diffractionKochinVector;

  }

  void Kochin::SetDiffractionKochin(unsigned int iwave, const Eigen::MatrixXcd &diffractionKochinMatrix) {

    // Setter of the diffraction Kochin function for all wave frequencies, all angles and a single wave direction.

    assert(iwave < m_HDB->GetWaveDirectionDiscretization().size());
    assert(diffractionKochinMatrix.rows() == m_nb_kochin_angle);
    assert(diffractionKochinMatrix.cols() == m_HDB->GetFrequencyDiscretization().size());
    m_kochin_diffraction[iwave] = diffractionKochinMatrix;

  }

  Eigen::MatrixXcd Kochin::GetDiffractionKochin(unsigned int iwave) const {

    // Getter of the diffraction Kochin function for all wave frequencies, all angles and a single wave direction.

    return m_kochin_diffraction[iwave];

  }

  void Kochin::SetDiffractionKochinDerivative(unsigned int iwave, unsigned int iw
                                            , const Eigen::VectorXcd &diffractionKochinDerivativeVector) {

    // Setter of the angular derivative of the diffraction Kochin function for a single wave frequency, all angles and a single wave direction.

    assert(iwave < m_HDB->GetWaveDirectionDiscretization().size());
    assert(iw < m_HDB->GetFrequencyDiscretization().size());
    assert(diffractionKochinDerivativeVector.rows() == m_nb_kochin_angle);
    assert(diffractionKochinDerivativeVector.cols() == 1);
    m_kochin_diffraction_derivate[iwave].col(iw) = diffractionKochinDerivativeVector;

  }

  void Kochin::SetDiffractionKochinDerivative(unsigned int iwave, const Eigen::MatrixXcd &diffractionKochinDerivativeMatrix) {

    // Setter of the angular derivative of the diffraction Kochin function for all wave frequencies, all angles and a single wave direction.

    assert(iwave < m_HDB->GetWaveDirectionDiscretization().size());
    assert(diffractionKochinDerivativeMatrix.rows() == m_nb_kochin_angle);
    assert(diffractionKochinDerivativeMatrix.cols() == m_HDB->GetFrequencyDiscretization().size());
    m_kochin_diffraction_derivate[iwave] = diffractionKochinDerivativeMatrix;

  }

  Eigen::MatrixXcd Kochin::GetDiffractionKochinDerivative(unsigned int iwave) const {

    // Getter of the angular derivative of the diffraction Kochin function for all wave frequencies, all angles and a single wave direction.

    return m_kochin_diffraction_derivate[iwave];

  }

  void Kochin::SetRadiationKochin(Body *Body, const Eigen::MatrixXcd &radiationKochinMatrix) {

    // Setter of the radiation Kochin function for a single body, a single dof, all wave frequencies and all angles.

    // Initialization.
    if (m_kochin_radiation.count(Body) == 0) {
      std::vector<Eigen::MatrixXcd> tempMat;
      tempMat.reserve(6);
      m_kochin_radiation.insert(std::make_pair(Body, tempMat));
    }

    assert(radiationKochinMatrix.rows() == m_nb_kochin_angle);
    assert(radiationKochinMatrix.cols() == m_HDB->GetFrequencyDiscretization().size());
    m_kochin_radiation.at(Body).push_back(radiationKochinMatrix);

  }

  void Kochin::SetRadiationKochin(Body *Body, unsigned int iw, unsigned int idof, const Eigen::MatrixXcd &radiationKochinMatrix) {

    // Setter of the radiation Kochin function for a single body, a single dof, a single wave frequency and all angles.

    // Initialization.
    if (m_kochin_radiation.count(Body) == 0) {
      std::vector<Eigen::MatrixXcd> tempMat;
      tempMat.reserve(6);
      m_kochin_radiation.insert(std::make_pair(Body, tempMat));
      for (int i = 0; i < 6; i++){
        auto tmp = Eigen::MatrixXcd(m_nb_kochin_angle, m_HDB->GetFrequencyDiscretization().size());
        m_kochin_radiation.at(Body).push_back(tmp);
      }
    }

    assert(radiationKochinMatrix.rows() == m_nb_kochin_angle);
    assert(radiationKochinMatrix.cols() == 1);
    m_kochin_radiation.at(Body)[idof].col(iw) = radiationKochinMatrix;

  }

  Eigen::MatrixXcd Kochin::GetRadiationKochin(Body *Body, unsigned int idof) const {

    // Getter of the radiation Kochin function for a single body, a single wave frequency, all dof and all angles.

    return m_kochin_radiation.at(Body)[idof];

  }

  void Kochin::SetRadiationKochinDerivative(Body *Body, const Eigen::MatrixXcd &radiationKochinDerivativeMatrix) {

    // Setter of the angular derivative of the radiation Kochin function for a single body, a single wave frequency, all dof and all angles.

    // Initialization.
    if (m_kochin_radiation_derivate.count(Body) == 0) {
      std::vector<Eigen::MatrixXcd> tempMat;
      tempMat.reserve(6);
      m_kochin_radiation_derivate.insert(std::make_pair(Body, tempMat));
    }

    assert(radiationKochinDerivativeMatrix.rows() == m_nb_kochin_angle);
    assert(radiationKochinDerivativeMatrix.cols() == m_HDB->GetFrequencyDiscretization().size());
    m_kochin_radiation_derivate.at(Body).push_back(radiationKochinDerivativeMatrix);

  }

  void Kochin::SetRadiationKochinDerivative(Body *Body, unsigned int iw, unsigned int idof, const Eigen::MatrixXcd &radiationKochinMatrix) {

    // Setter of the radiation Kochin angular derivative of the function for a single body, a single dof, a single wave frequency and all angles.

    // Initialization.
    if (m_kochin_radiation_derivate.count(Body) == 0) {
      std::vector<Eigen::MatrixXcd> tempMat;
      tempMat.reserve(6);
      m_kochin_radiation_derivate.insert(std::make_pair(Body, tempMat));
      for (int i = 0; i < 6; i++){
        auto tmp = Eigen::MatrixXcd(m_nb_kochin_angle, m_HDB->GetFrequencyDiscretization().size());
        m_kochin_radiation_derivate.at(Body).push_back(tmp);
      }
    }

    assert(radiationKochinMatrix.rows() == m_nb_kochin_angle);
    assert(radiationKochinMatrix.cols() == 1);
    m_kochin_radiation_derivate.at(Body)[idof].col(iw) = radiationKochinMatrix;

  }

  Eigen::MatrixXcd Kochin::GetRadiationKochinDerivative(Body *Body, unsigned int idof) const {

    // Getter of the angular derivative of the radiation Kochin function for a single body, a single wave frequency, all dof and all angles.

    return m_kochin_radiation_derivate.at(Body)[idof];

  }

}