//
// Created by lletourn on 27/02/20.
//

#include "Mask.h"

namespace hdb5_io {

  void Mask::SetMask(mathutils::Vector6d<bool> mask) {

    for (unsigned int i = 0; i < 6; i++) { assert(mask(i) == 0 || mask(i) == 1); }

    m_nbDOF = 0;
    for (unsigned int i = 0; i < 6; i++) {
      if (mask(i)) {
        m_mask(i) = true;
        m_DOFs.emplace_back(i);
        m_nbDOF++;
      } else {
        m_mask(i) = false;
      }
    }

//    m_nbDOF = (unsigned int) mask.sum();

    m_matrix = Eigen::MatrixXd::Zero(6, m_nbDOF);
    unsigned int j = 0;
    for (unsigned int i = 0; i < 6; i++) {
      if (mask(i)) {
        m_matrix(i, j) = 1.;
        j += 1;
      }
    }
  }

  mathutils::Vector6d<bool> Mask::GetMask() const {
    return m_mask;
  }

  mathutils::MatrixMN<double> Mask::GetMatrix() const {
    return m_matrix;
  }

  std::vector<unsigned int> Mask::GetListDOF() const {

    std::vector<unsigned int> DOFlist;

    for (const auto &dof : m_DOFs) {
      DOFlist.push_back(dof.GetIndex());
    }

    return DOFlist;
  }

  Mask Mask::operator||(const Mask &otherMask) const {
    Mask newMask;

    auto thisVectorMask = this->GetMask();

    auto otherVectorMask = otherMask.GetMask();

    newMask.SetMask(thisVectorMask||otherVectorMask);

    return newMask;
  }

  Mask Mask::operator&&(const Mask &otherMask) const {
    Mask newMask;

    auto thisVectorMask = this->GetMask();

    auto otherVectorMask = otherMask.GetMask();

    newMask.SetMask(thisVectorMask&&otherVectorMask);

    return newMask;
  }


} // namespace hdb5_io
