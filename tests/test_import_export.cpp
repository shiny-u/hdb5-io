//
// Created by lletourn on 27/02/20.
//

#include "gtest/gtest.h"
#include "hdb5_io/hdb5_io.h"

using namespace hdb5_io;

TEST(hdb5_io, import_export) {

  auto HDB = import_HDB("Boxbarge_Vertices_353_Faces_652.hdb5");

  // Add Poles and residues.
  Eigen::MatrixXi nb_real(6, 6), nb_cc(6, 6);
  nb_real << 1, 0, 1, 2, 3, 6,
      1, 1, 0, 0, 0, 0,
      1, 2, 3, 1, 1, 0,
      3, 4, 2, 0, 0, 0,
      2, 1, 1, 0, 1, 0,
      2, 1, 1, 0, 1, 3;
  nb_cc << 0, 1, 6, 6, 5, 6,
      5, 4, 6, 5, 5, 4,
      4, 5, 6, 6, 5, 6,
      5, 4, 6, 5, 5, 4,
      4, 5, 6, 6, 5, 6,
      5, 4, 6, 5, 5, 4;

  for (int idof = 0; idof < 6; idof++) {

    std::vector<PoleResidue> modalCoeff;
    for (int iforce = 0; iforce < 6; iforce++) {

      // Real poles and residues.
      int n_pole_real = nb_real(idof, iforce);
      Eigen::VectorXd poles(n_pole_real), residues(n_pole_real);
      poles.setRandom();
      residues.setRandom();

      // Complex poles and residues.
      int n_pole_cc = nb_cc(idof, iforce);
      Eigen::VectorXcd cplxPoles(n_pole_cc), cplxResidues(n_pole_cc);
      cplxPoles.setRandom();
      cplxResidues.setRandom();

      PoleResidue pair;
      for (int i = 0; i < n_pole_real; i++) {
        pair.AddPoleResidue(poles(i), residues(i));
      }
      for (int i = 0; i < n_pole_cc; i++) {
        pair.AddPoleResidue(cplxPoles(i), cplxResidues(i));
      }
      modalCoeff.emplace_back(pair);

    }
    HDB->GetBody(0)->AddModalCoefficients(HDB->GetBody(0), modalCoeff);
  }

  // Add Kochin angular step.
  double kochin_step = 1.;
  int nbDirKochin = HDB->GetWaveDirectionDiscretization().size();
  auto kochin = std::make_shared<Kochin>(HDB.get(), kochin_step * MU_PI_180, nbDirKochin);
  int nAngles = kochin->GetNbKochinAngles();
  kochin->SetWaveDirectionKochin(HDB->GetWaveDirectionDiscretization());

  // Add diffraction Kochin functions.
  for (unsigned int iwaveDir = 0; iwaveDir < nbDirKochin; ++iwaveDir) {

    // Function.
    Eigen::MatrixXcd kochin_diffraction(nAngles, HDB->GetFrequencyDiscretization().size());
    kochin_diffraction.setRandom();
    kochin->SetDiffractionKochin(iwaveDir, kochin_diffraction);

    // Derivative.
    Eigen::MatrixXcd kochin_diffraction_derivative(nAngles, HDB->GetFrequencyDiscretization().size());
    kochin_diffraction_derivative.setRandom();
    kochin->SetDiffractionKochinDerivative(iwaveDir, kochin_diffraction_derivative);

  }

  // Add radiation Kochin functions.
  for (unsigned int idof = 0; idof < 6; ++idof) {

    // Function.
    Eigen::MatrixXcd kochin_radiation(nAngles, HDB->GetFrequencyDiscretization().size());
    kochin_radiation.setRandom();
    kochin->SetRadiationKochin(HDB->GetBody(0), kochin_radiation);

    // Derivative.
    Eigen::MatrixXcd kochin_radiation_derivative(nAngles, HDB->GetFrequencyDiscretization().size());
    kochin_radiation_derivative.setRandom();
    kochin->SetRadiationKochinDerivative(HDB->GetBody(0), kochin_radiation_derivative);

  }

  HDB->SetKochin(kochin);

  // Add Helios commit hash.
  HDB->SetSolver("Helios");
  HDB->SetNormalizedVersionString(git::GetNormalizedVersionString());

  // Expert numerical parameters.
  HDB->SetExpertParameters();
  HDB->SetSurfaceIntegrationOrder(4);
  HDB->SetGreenFunction("newman");
  HDB->SetCrmax(10);

  // Horizontal position.
  HDB->GetBody(0)->SetHorizontalPositionInWorld(Eigen::Vector3d(4, 5, 90));

  // Computation point.
  HDB->GetBody(0)->SetComputationPointInBodyFrame(Eigen::Vector3d(1, 2, 3));

  // Wave reference point.
  HDB->GetBody(0)->SetWaveReferencePointInBodyFrame(Eigen::Vector2d(4, 5));

  export_HDB("out.hdb5", HDB.get());

  auto HDB_2 = import_HDB("out.hdb5");

  export_HDB("out_2.hdb5", HDB_2.get());

  std::cout << "Import-export test: ok!" << std::endl;

}
