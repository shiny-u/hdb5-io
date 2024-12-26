//
// Created by lletourn on 07/04/20.
//

#ifndef HDB5_IO_WAVEDRIFT_H
#define HDB5_IO_WAVEDRIFT_H

#include <memory>
#include "MathUtils/LookupTable2D.h"
#include "MathUtils/Matrix.h"
#include "MathUtils/VectorN.h"

namespace hdb5_io {

  class WaveDrift {

   public:

    WaveDrift();

    void SetSymmetries(bool symmetry_X, bool symmetry_Y);

    std::vector<bool> GetSymmetries() const;

    void SetFrequencies(const mathutils::VectorN<double> &frequencies);

    void SetWaveDirections(const mathutils::VectorN<double> &angles);

    /// Adding new data with wave drift coefficients
    /// \param name Name of the data
    /// \param coeffs >Wave drift coefficient values
    void AddData(const std::string &name, const std::vector<double> &coeffs);

    double Eval(const std::string &name, double frequency, double angle) const;

    void SetKochinStep(const double &kochin_step);

    double GetKochinStep() const;

//    double Eval(const std::string &name, std::vector<double> frequencies, std::vector<double> angles) const {
//      return m_data->Eval(name, frequencies, angles);
//    }

    bool HasSurge() const { return m_data->HasSerie("surge"); }

    bool HasSway() const { return m_data->HasSerie("sway"); }

    bool HasHeave() const { return m_data->HasSerie("heave"); }

    bool HasPitch() const { return m_data->HasSerie("pitch"); }

    bool HasRoll() const { return m_data->HasSerie("roll"); }

    bool HasYaw() const { return m_data->HasSerie("yaw"); }

   protected:

    bool m_symmetry_X;
    bool m_symmetry_Y;

    double m_kochin_step; /// Kochin angle step in degrees.

    std::unique_ptr<mathutils::LookupTable2d<double>> m_data;

  };

}

#endif //HDB5_IO_WAVEDRIFT_H
