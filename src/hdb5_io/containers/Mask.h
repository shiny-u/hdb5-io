//
// Created by lletourn on 27/02/20.
//

#ifndef HDB5_IO_MASK_H
#define HDB5_IO_MASK_H

#include "vector"

#include "MathUtils/Matrix.h"
#include "MathUtils/Vector3d.h"
#include "MathUtils/Vector6d.h"

namespace hdb5_io {

  class DOF {

   public:
    enum TYPE {
      LINEAR,
      ANGULAR
    };

   private:
    TYPE m_type;
    mathutils::Vector3d<double> m_direction;
//    Position m_position;
    bool m_active = true;
    unsigned int m_index;

   public:

    DOF(unsigned int index) : m_index(index) {
      switch (index) {
        case 0: {
          m_type = LINEAR;
          m_direction = {1,0,0};
          break;
        }
        case 1: {
          m_type = LINEAR;
          m_direction = {0,1,0};
          break;
        }
        case 2: {
          m_type = LINEAR;
          m_direction = {0,0,1};
          break;
        }
        case 3: {
          m_type = ANGULAR;
          m_direction = {1,0,0};
          break;
        }
        case 4: {
          m_type = ANGULAR;
          m_direction = {0,1,0};
          break;
        }
        case 5: {
          m_type = ANGULAR;
          m_direction = {0,0,1};
          break;
        }
        default: {
          std::cerr<<"wrong dof index"<<std::endl;
          break;
        }
      }

    }

    void SetTypeLINEAR() { m_type = LINEAR; }

    void SetTypeANGULAR() { m_type = ANGULAR; }

    unsigned int GetIndex() const { return m_index;}

    TYPE GetType() const { return m_type; }

    void SetDirection(mathutils::Vector3d<double> &direction) { m_direction = direction; }

    mathutils::Vector3d<double> GetDirection() const { return m_direction; }

//    void SetPointPosition(Position &position) { m_position = position; }
//
//    Position GetPointPosition() const { return m_position; }

    // TODO : flag pour le couplage FrBody/FrBEMBody (a voir si conservation necessaire)
    void Activate() { m_active = true; }

    void Deactivate() { m_active = false; }

    bool IsActive() const { return m_active; }

  };

  /**
  * \class Mask
  * \brief Class for the force and motions masks
  */
  class Mask {

   private:
    unsigned int m_nbDOF;                   ///< Number of degrees of freedom of the body
    std::vector<DOF> m_DOFs;                ///< List of degrees of freedom of the body
    mathutils::Vector6d<bool> m_mask;       ///< Mask applied on the degrees of freedom
    mathutils::MatrixMN<double> m_matrix;   ///<

   public:
    /// Default constructor of the DOF Mask
    Mask() {}

    /// Define the DOF mask from vector
    /// \param mask DOF Mask vector
    void SetMask(mathutils::Vector6d<bool> mask);

    /// Return the DOF Mask
    /// \return DOF Mask
    mathutils::Vector6d<bool> GetMask() const;

    /// Return the mask for a specific degree of freedom
    /// \param imode Degree of freedom
    /// \return True if the mask is active, False otherwise
    bool GetMask(unsigned int imode) const { return m_mask[imode]; }

    ///
    /// \return
    mathutils::MatrixMN<double> GetMatrix() const;

    /// Return the number of activated DOF in the mask
    /// \return Number of activated DOF
    unsigned int GetNbDOF() const { return m_nbDOF; }

    /// Return the list of activated DOF
    /// \return List of activated DOF
    std::vector<unsigned int> GetListDOF() const;

    std::vector<DOF> GetDOFs() const { return m_DOFs;}

    Mask operator||(const Mask &otherMask) const;

    Mask operator&&(const Mask &otherMask) const;

  };


} // namespace hdb5_io


#endif //HDB5_IO_MASK_H
