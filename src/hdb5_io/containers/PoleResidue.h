//
// Created by lletourn on 16/04/20.
//

#ifndef HDB5_IO_POLERESIDUE_H
#define HDB5_IO_POLERESIDUE_H

#include <complex>
#include <vector>

#include <Eigen/Dense>

namespace hdb5_io {


  /**
  * \class PoleResiduePair
  * \brief Class for storing the paired modal coefficients : pole and residue
   * Templated depending if the pole and residue are real or complex.
  */
  //TODO:: changed to std::pair
  template<typename T>
  class PoleResiduePair {

   public:
    PoleResiduePair(const T &pole, const T &residue) : m_pole(pole), m_residue(residue) {}

    inline const T &pole() const {
      return m_pole;
    }

    inline const T &residue() const {
      return m_residue;
    }

    inline const T &state() const {
      return m_state;
    }

    inline void state(T state) {
      m_state = state;
    }

    bool operator==(const PoleResiduePair<T> otherPair) const {
      return m_pole == otherPair.m_pole && m_residue == otherPair.m_residue;
    }

   private:
    T m_pole;
    T m_residue;
    T m_state;
  };

  using RealPoleResiduePair = PoleResiduePair<double>;
  using CCPoleResiduePair = PoleResiduePair<std::complex<double>>;
//  using RealPoleResiduePair = std::pair<double, double>;
//  using CCPoleResiduePair = std::pair<std::complex<double>,std::complex<double>>;

  /**
  * \class PoleResidue
  * \brief Class for storing the modal coefficients : poles and residues (real and complex ones)
  */
  class PoleResidue {

   public:

    PoleResidue() = default;

    unsigned int nb_real_poles() const;

    unsigned int nb_cc_poles() const;

    unsigned int order() const;

    void AddPoleResidue(const double &pole, const double &residue);

    void AddPoleResidue(const std::complex<double> &pole, const std::complex<double> &residue);

    std::vector<RealPoleResiduePair> GetRealPairs() const;

    std::vector<CCPoleResiduePair> GetComplexPairs() const;

    std::vector<double> GetRealPoles() const;

    std::vector<double> GetRealResidues() const;

    Eigen::VectorXcd GetResidues() const;

    Eigen::VectorXcd GetPoles() const;

    Eigen::VectorXcd GetStates() const;

    void SetStates(Eigen::VectorXcd states);


    /// Get the complex poles in a matrix form, where the real part is in the first row and the imaginary part in the
    /// second row. For writing purpose mostly
    Eigen::MatrixXd GetComplexPoles() const;

    /// Get the complex residues in a matrix form, where the real part is in the first row and the imaginary part in the
    /// second row. For writing purpose mostly
    Eigen::MatrixXd GetComplexResidues() const;

   private:

    std::vector<RealPoleResiduePair> m_real_pairs;  ///< container of the real poles and residues
    std::vector<CCPoleResiduePair> m_cc_pairs;      ///< container of the complex poles and residues

  };

} // end namespace hdb5_io
#endif //HDB5_IO_POLERESIDUE_H
