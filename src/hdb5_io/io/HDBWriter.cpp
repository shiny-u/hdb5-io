//
// Created by lletourn on 15/04/20.
//

#include <Eigen/Dense>
#include <highfive/H5Easy.hpp>

#include "HDBWriter.h"

#include "hdb5_io/containers/HydrodynamicDataBase.h"
#include "hdb5_io/containers/Body.h"
#include "hdb5_io/containers/WaveDrift.h"
#include "hdb5_io/containers/Kochin.h"
#include "hdb5_io/containers/PoleResidue.h"

#include "hdb5_io/version.h"

namespace hdb5_io {


  void HDBWriter::Write(const std::string &filename) const {

    // HDB file.
    HighFive::File file(filename, HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate);

    // HDB basic data (version, date, gravity constant, density, etc.).
    WriteHDBBasics(file);

    // Wave frequency and wave directions.
    WriteDiscretizations(file);

    // Symmetries.
    if (m_hdb->GetSymmetries()) {
      WriteSymmetries(file);
    }

    auto bodies = file.createGroup("Bodies");

    for (unsigned int i = 0; i < m_hdb->GetNbBodies(); i++) {

      auto bodyGroup = bodies.createGroup("Body_" + std::to_string(i));
      auto body = m_hdb->GetBody(i);
      auto bodyPath = "Bodies/Body_" + std::to_string(i);

      // Body basic data (index, name, position, mass, etc.).
      WriteBodyBasics(file, "Bodies/Body_" + std::to_string(i), body);

      #ifdef MESH_SUPPORT
      // Body mesh.
      WriteMesh(file, bodyPath, body);
      #endif

      // Diffraction loads.
      WriteExcitation(excitationType::Diffraction, file, bodyPath + "/Excitation/Diffraction", body);

      // x-derivative of the diffraction loads.
      if (m_hdb->GetIsXDerivative()) {
        WriteExcitation(excitationType::DiffractionXDerivative, file, bodyPath + "/Excitation/DiffractionXDerivative",
                        body);
      }

      // Froude-Krylov loads.
      WriteExcitation(excitationType::Froude_Krylov, file, bodyPath + "/Excitation/FroudeKrylov", body);

      // x-derivative of the Froude-Krylov loads.
      if (m_hdb->GetIsXDerivative()) {
        WriteExcitation(excitationType::Froude_KrylovXDerivative, file,
                        bodyPath + "/Excitation/FroudeKrylovXDerivative", body);
      }

      // Added mass, damping, IRF, poles and residues and the x-derivatives.
      WriteRadiation(file, bodyPath + "/Radiation", body);

      // RAOs.
      if (body->HasRAO()) {
        WriteRAO(file, bodyPath + "/RAO", body);
      }

      // Mean wave drift loads.
      if (body->GetWaveDrift()) {
        WriteWaveDrift(file, bodyPath + "/WaveDrift", body);
      }
    }

    if (m_hdb->GetKochin()) {
      WriteKochin(file);
    }

    // Wave field.
    if (m_hdb->GetWaveField()) {
      WriteWaveField(file);
    }

    // Vector fitting parameters.
    if (m_hdb->GetVF()) {
      WriteVF(file);
    }

    // Expert numerical parameters.
    if (m_hdb->GetExpertParameters()) {
      WriteExpertParameters(file);
    }

  }


  void HDBWriter::WriteHDBBasics(HighFive::File &file) const {

    auto creationDate = m_hdb->GetCreationDate();
    HighFive::DataSet dataSet = file.createDataSet<std::string>("CreationDate",
                                                                HighFive::DataSpace::From(creationDate));
    dataSet.write(creationDate);
    dataSet.createAttribute<std::string>("Description", "Date of the creation of this database.");

    auto version = HDB_NUM_VERSION;
    dataSet = file.createDataSet<double>("Version", HighFive::DataSpace::From(version));
    dataSet.write(version);
    dataSet.createAttribute<std::string>("Description", "Version of the hdb5 output file.");

    auto solver = m_hdb->GetSolver();
    dataSet = file.createDataSet<std::string>("Solver", HighFive::DataSpace::From(solver));
    dataSet.write(solver);
    dataSet.createAttribute<std::string>("Description",
                                         "Hydrodynamic solver used for computing the hydrodynamic database.");

    if (m_hdb->GetSolver() == "Helios") {
      if (m_hdb->IsNormalizedVersionString()) {
        auto commit_hash = m_hdb->GetNormalizedVersionString();
        dataSet = file.createDataSet<std::string>("NormalizedCommitHash", HighFive::DataSpace::From(commit_hash));
        dataSet.write(commit_hash);
        dataSet.createAttribute<std::string>("Description", "Tag - Commit hash - Branch - Date.");
      }
    }

    auto nbody = m_hdb->GetNbBodies();
    dataSet = file.createDataSet<int>("NbBody", HighFive::DataSpace::From(nbody));
    dataSet.write(nbody);
    dataSet.createAttribute<std::string>("Description", "Number of hydrodynamic bodies.");

    auto normalizationLength = m_hdb->GetNormalizationLength();
    dataSet = file.createDataSet<double>("NormalizationLength", HighFive::DataSpace::From(normalizationLength));
    dataSet.write(normalizationLength);
    dataSet.createAttribute<std::string>("Description", "Normalization length.");
    dataSet.createAttribute<std::string>("Unit", "m");

    auto gravityAcceleration = m_hdb->GetGravityAcceleration();
    dataSet = file.createDataSet<double>("GravityAcc", HighFive::DataSpace::From(gravityAcceleration));
    dataSet.write(gravityAcceleration);
    dataSet.createAttribute<std::string>("Description", "Gravity acceleration.");
    dataSet.createAttribute<std::string>("Unit", "m/s**2");

    auto waterDensity = m_hdb->GetWaterDensity();
    dataSet = file.createDataSet<double>("WaterDensity", HighFive::DataSpace::From(waterDensity));
    dataSet.write(waterDensity);
    dataSet.createAttribute<std::string>("Description", "Water Density.");
    dataSet.createAttribute<std::string>("Unit", "kg/m**3");

    auto waterDepth = m_hdb->GetWaterDepth();
    dataSet = file.createDataSet<double>("WaterDepth", HighFive::DataSpace::From(waterDepth));
    dataSet.write(waterDepth);
    dataSet.createAttribute<std::string>("Description",
                                         "Water depth: 0 for infinite depth and positive for finite depth.");
    dataSet.createAttribute<std::string>("Unit", "m");
  }

  void HDBWriter::WriteDiscretizations(HighFive::File &file) const {

    auto discretizations = file.createGroup("Discretizations");
    H5Easy::dump(file, "Discretizations/Frequency",
                 static_cast<Eigen::Matrix<double, Eigen::Dynamic, 1>> (m_hdb->GetFrequencyDiscretization()));

    // Conversion in degrees.
    H5Easy::dump(file, "Discretizations/WaveDirection",
                 static_cast<Eigen::Matrix<double, Eigen::Dynamic, 1>> (m_hdb->GetWaveDirectionDiscretization()
                                                                        * MU_180_PI));
    H5Easy::dump(file, "Discretizations/Time",
                 static_cast<Eigen::Matrix<double, Eigen::Dynamic, 1>> (m_hdb->GetTimeDiscretization()));

  }

  void HDBWriter::WriteSymmetries(HighFive::File &file) const {

    auto Symmetries = file.createGroup("Symmetries");
    H5Easy::dump(file, "Symmetries/Bottom", static_cast<int> (m_hdb->GetSymBottom()));
    H5Easy::dump(file, "Symmetries/xOz", static_cast<int> (m_hdb->GetSymXOZ()));
    H5Easy::dump(file, "Symmetries/yOz", static_cast<int> (m_hdb->GetSymYOZ()));

  }

  void HDBWriter::WriteBodyBasics(HighFive::File &file, const std::string &path, Body *body) const {

    // This method writes the basic data for a body.

    auto bodyGroup = file.getGroup(path);

    // index.
    auto dataSet = bodyGroup.createDataSet<unsigned int>("ID", HighFive::DataSpace::From(body->GetID()));
    dataSet.write(body->GetID());
    dataSet.createAttribute<std::string>("Description", "Body index");

    // Name.
    dataSet = bodyGroup.createDataSet<std::string>("BodyName", HighFive::DataSpace::From(body->GetName()));
    dataSet.write(body->GetName());
    dataSet.createAttribute<std::string>("Description", "Body name");

    // Horizontal in the world frame.
    auto BodyPosition = body->GetHorizontalPositionInWorld();
    H5Easy::dump(file, path + "/HorizontalPosition/x", static_cast<double> (BodyPosition(0)));
    H5Easy::dump(file, path + "/HorizontalPosition/y", static_cast<double> (BodyPosition(1)));
    H5Easy::dump(file, path + "/HorizontalPosition/psi", static_cast<double> (BodyPosition(2))); // Degrees.

    // Computation point in the body frame.
    H5Easy::dump(file, path + "/ComputationPoint",
                 static_cast<Eigen::Vector3d> (body->GetComputationPointInBodyFrame()));

    // Wave reference point in the body frame.
    H5Easy::dump(file, path + "/WaveReferencePoint",
                 static_cast<Eigen::Vector2d> (body->GetWaveReferencePointInBodyFrame()));

    // Mask.
    auto ExcitationGroup = bodyGroup.createGroup("Excitation");
    ExcitationGroup.createGroup("Diffraction");
    ExcitationGroup.createGroup("FroudeKrylov");
    //TODO : move mask in Excitation folder once it has be done in HDB5Tool too
    H5Easy::dump(file, path + "/Mask/ForceMask",
                 static_cast<Eigen::Matrix<bool, 6, 1>> (body->GetForceMask().GetMask()));

    // Inertia matrix.
    if (body->HasInertia()) {
      bodyGroup.createGroup("Inertia");
      H5Easy::dump(file, path + "/Inertia/InertiaMatrix",
                   static_cast<Eigen::Matrix<double, 6, 6>> (body->GetInertiaMatrix()));
      bodyGroup.getDataSet("Inertia/InertiaMatrix").createAttribute<std::string>("Description", "Inertia matrix.");
    }

    // Hydrostatic matrix.
    if (body->HasHydrostatic()) {
      bodyGroup.createGroup("Hydrostatic");
      Eigen::Matrix<double, 6, 6> stiffnessMatrix;
      stiffnessMatrix.setZero();
      stiffnessMatrix.block<3, 3>(2, 2) = body->GetHydrostaticStiffnessMatrix();
      H5Easy::dump(file, path + "/Hydrostatic/StiffnessMatrix", stiffnessMatrix);
      bodyGroup.getDataSet("Hydrostatic/StiffnessMatrix").createAttribute<std::string>("Description",
                                                                                       "Hydrostatic stiffness matrix.");
    }

    // Mooring matrix.
    if (body->HasMooring()) {
      bodyGroup.createGroup("Mooring");
      H5Easy::dump(file, path + "/Mooring/MooringMatrix",
                   static_cast<Eigen::Matrix<double, 6, 6>> (body->GetMooringMatrix()));
      bodyGroup.getDataSet("Mooring/MooringMatrix").createAttribute<std::string>("Description", "Mooring matrix.");
    }

    // Damping matrix.
    if (body->HasDamping()) {
      bodyGroup.createGroup("LinearDamping");
      H5Easy::dump(file, path + "/LinearDamping/DampingMatrix",
                   static_cast<Eigen::Matrix<double, 6, 6>> (body->GetDampingMatrix()));
      bodyGroup.getDataSet("LinearDamping/DampingMatrix").createAttribute<std::string>("Description",
                                                                                       "Linear damping matrix.");
    }

  }

#ifdef MESH_SUPPORT

  void HDBWriter::WriteMesh(HighFive::File &file, const std::string &path, Body *body) const {

    auto bodyGroup = file.getGroup(path);

    bodyGroup.createGroup("Mesh");
    H5Easy::dump(file, path + "/Mesh/NbFaces", body->GetMesh()->n_faces());
    H5Easy::dump(file, path + "/Mesh/Faces", body->GetMesh()->GetFaces());
    H5Easy::dump(file, path + "/Mesh/NbVertices", body->GetMesh()->n_vertices());
    H5Easy::dump(file, path + "/Mesh/Vertices", body->GetMesh()->GetVertices());

  }

#endif

  void HDBWriter::WriteExcitation(HDBWriter::excitationType type, HighFive::File &HDF5_file, const std::string &path,
                                  Body *body) const {

    auto angles = m_hdb->GetWaveDirectionDiscretization();

    for (unsigned int iwaveDir = 0; iwaveDir < angles.size(); ++iwaveDir) {

      auto anglePath = path + "/Angle_" + std::to_string(iwaveDir);

      auto angle = angles(iwaveDir);

      Eigen::MatrixXcd coeff;
      switch (type) {
        case Diffraction : {
          coeff = body->GetDiffraction(iwaveDir);
          break;
        }
        case DiffractionXDerivative : {
          coeff = body->GetXDerivativeDiffraction(iwaveDir);
          break;
        }
        case Froude_Krylov : {
          coeff = body->GetFroudeKrylov(iwaveDir);
          break;
        }
        case Froude_KrylovXDerivative : {
          coeff = body->GetXDerivativeFroudeKrylov(iwaveDir);
          break;
        }
      }
      auto angleGroup = HDF5_file.createGroup(anglePath);
      H5Easy::dump(HDF5_file, anglePath + "/Angle", angle * MU_180_PI);
      angleGroup.getDataSet("Angle").createAttribute<std::string>("Description", "Wave direction.");
      angleGroup.getDataSet("Angle").createAttribute<std::string>("Unit", "deg");

      H5Easy::dump(HDF5_file, anglePath + "/RealCoeffs", static_cast<Eigen::MatrixXd>(coeff.real()));
      angleGroup.getDataSet("RealCoeffs").createAttribute<std::string>("Description",
                                                                       "Real part of the Froude-Krylov loads on body " +
                                                                       std::to_string(body->GetID()) +
                                                                       " for a wave direction of " + std::to_string(
                                                                           angle * MU_180_PI) +
                                                                       " deg.");
      angleGroup.getDataSet("RealCoeffs").createAttribute<std::string>("Unit", "N/m");

      H5Easy::dump(HDF5_file, anglePath + "/ImagCoeffs", static_cast<Eigen::MatrixXd>(coeff.imag()));
      angleGroup.getDataSet("ImagCoeffs").createAttribute<std::string>("Description",
                                                                       "Imaginary part of the Froude-Krylov loads on body " +
                                                                       std::to_string(body->GetID()) +
                                                                       " for a wave direction of " + std::to_string(
                                                                           angle * MU_180_PI) +
                                                                       " deg.");
      angleGroup.getDataSet("ImagCoeffs").createAttribute<std::string>("Unit", "N/m");

    }

  }

  void HDBWriter::WriteRadiation(HighFive::File &HDF5_file, const std::string &path, Body *body) const {

//    auto frequencies = m_hdb->GetFrequencyDiscretization();
    auto time = m_hdb->GetTimeDiscretization();

    for (unsigned int ibodyMotion = 0; ibodyMotion < m_hdb->GetNbBodies(); ++ibodyMotion) {

      auto bodyMotion = m_hdb->GetBody(ibodyMotion);
      auto bodyMotionPath = path + "/BodyMotion_" + std::to_string(ibodyMotion);
      auto bodyMotionGroup = HDF5_file.createGroup(bodyMotionPath);
      bodyMotionGroup.createAttribute<std::string>("Description", "Hydrodynamic coefficients for motion of body " +
                                                                  std::to_string(bodyMotion->GetID()) +
                                                                  " that radiates waves and  generate force on body " +
                                                                  std::to_string(body->GetID()) + ".");

      // Writing the infinite frequency added mass matrix for the body.
      H5Easy::dump(HDF5_file, bodyMotionPath + "/InfiniteAddedMass",
                   static_cast<Eigen::MatrixXd>(body->GetInfiniteAddedMass(bodyMotion)));
      auto InfiniteAddedMass = bodyMotionGroup.getDataSet("InfiniteAddedMass");
      InfiniteAddedMass.createAttribute<std::string>("Description",
                                                     "Infinite added mass matrix that modifies the apparent mass of body " +
                                                     std::to_string(bodyMotion->GetID()) +
                                                     " from acceleration of body  " +
                                                     std::to_string(body->GetID()) + ".");

      if (m_hdb->GetIsXDerivative()) {
        H5Easy::dump(HDF5_file, bodyMotionPath + "/InfiniteAddedMassXDerivative",
                     static_cast<Eigen::MatrixXd>(body->GetXDerivativeInfiniteAddedMass(bodyMotion)));
        auto InfiniteAddedMassXDerivative = bodyMotionGroup.getDataSet("InfiniteAddedMassXDerivative");
        InfiniteAddedMassXDerivative.createAttribute<std::string>("Description",
                                                                  "x-derivative of the infinite added mass matrix that modifies the apparent mass of body " +
                                                                  std::to_string(bodyMotion->GetID()) +
                                                                  " from acceleration of body  " +
                                                                  std::to_string(body->GetID()) + ".");
      }

      // Writing the zero frequency added mass matrix for the body.
      if (body->HasZeroFreqAddedMass(bodyMotion)) {
        H5Easy::dump(HDF5_file, bodyMotionPath + "/ZeroFreqAddedMass",
                     static_cast<Eigen::MatrixXd>(body->GetZeroFreqAddedMass(bodyMotion)));
        auto ZeroFreqAddedMass = bodyMotionGroup.getDataSet("ZeroFreqAddedMass");
        ZeroFreqAddedMass.createAttribute<std::string>("Description",
                                                       "Zero frequency added mass matrix that modifies the apparent mass of body " +
                                                       std::to_string(bodyMotion->GetID()) +
                                                       " from acceleration of body  " +
                                                       std::to_string(body->GetID()) + ".");
      }

      if (m_hdb->GetIsXDerivative()) {
        if (body->HasZeroFreqAddedMass(bodyMotion)) {
          H5Easy::dump(HDF5_file, bodyMotionPath + "/ZeroFreqAddedMassXDerivative",
                       static_cast<Eigen::MatrixXd>(body->GetXDerivativeZeroFreqAddedMass(bodyMotion)));
          auto ZeroFreqAddedMassXDerivative = bodyMotionGroup.getDataSet("ZeroFreqAddedMassXDerivative");
          ZeroFreqAddedMassXDerivative.createAttribute<std::string>("Description",
                                                                    "x-derivative of the zero frequency added mass matrix that modifies the apparent mass of body " +
                                                                    std::to_string(bodyMotion->GetID()) +
                                                                    " from acceleration of body  " +
                                                                    std::to_string(body->GetID()) + ".");
        }
      }

      // Writing the radiation mask matrix for the body.
      H5Easy::dump(HDF5_file, bodyMotionPath + "/RadiationMask",
                   static_cast<Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>>(body->GetRadiationMask(
                       bodyMotion)));
      auto RadiationMask = bodyMotionGroup.getDataSet("RadiationMask");
      RadiationMask.createAttribute<std::string>("Description", "Radiation mask of body " +
                                                                std::to_string(bodyMotion->GetID()) +
                                                                " from acceleration of body  " +
                                                                std::to_string(body->GetID()) + ".");

      // Writing the added mass matrix for the body.
      auto AddedMassGroup = bodyMotionGroup.createGroup("AddedMass");
      AddedMassGroup.createAttribute<std::string>("Description", "Added mass coefficients for acceleration of body " +
                                                                 std::to_string(bodyMotion->GetID()) +
                                                                 " that radiates waves and  generate force on body " +
                                                                 std::to_string(body->GetID()) + ".");
      for (unsigned int imotion = 0; imotion < 6; imotion++) {
        H5Easy::dump(HDF5_file, bodyMotionPath + "/AddedMass/DOF_" + std::to_string(imotion),
                     body->GetAddedMass(bodyMotion, imotion));
        auto DOF = AddedMassGroup.getDataSet("DOF_" + std::to_string(imotion));
        DOF.createAttribute<std::string>("Description", "Added mass coefficients for an acceleration of body " +
                                                        std::to_string(bodyMotion->GetID()) +
                                                        " and force on body " +
                                                        std::to_string(body->GetID()) + ".");
        //TODO : complete units, depending on DOF, bodyMotion, etc.
        DOF.createAttribute<std::string>("Unit", "");
      }

      // Writing the radiation damping matrix for the body.
      auto RadiationDampingGroup = bodyMotionGroup.createGroup("RadiationDamping");
      RadiationDampingGroup.createAttribute<std::string>("Description", "Damping coefficients for velocity of body " +
                                                                        std::to_string(bodyMotion->GetID()) +
                                                                        " that radiates waves and generates forces on body " +
                                                                        std::to_string(body->GetID()) + ".");
      for (unsigned int imotion = 0; imotion < 6; imotion++) {
        H5Easy::dump(HDF5_file, bodyMotionPath + "/RadiationDamping/DOF_" + std::to_string(imotion),
                     body->GetRadiationDamping(bodyMotion, imotion));
        auto DOF = RadiationDampingGroup.getDataSet("DOF_" + std::to_string(imotion));
        DOF.createAttribute<std::string>("Description", "Wave damping coefficients for an acceleration of body " +
                                                        std::to_string(bodyMotion->GetID()) +
                                                        " and force on body " +
                                                        std::to_string(body->GetID()) + ".");
        //TODO : complete units, depending on DOF, bodyMotion, etc.
        DOF.createAttribute<std::string>("Unit", "");
      }

      // Writing of the x-derivative of the radiation coefficients.
      if (m_hdb->GetIsXDerivative()) {
        auto AddedMassXDerivativeGroup = bodyMotionGroup.createGroup("AddedMassXDerivative");
        AddedMassXDerivativeGroup.createAttribute<std::string>("Description",
                                                               "x-derivative of the added mass coefficients for acceleration of body " +
                                                               std::to_string(bodyMotion->GetID()) +
                                                               " that radiates waves and  generate force on body " +
                                                               std::to_string(body->GetID()) + ".");
        for (unsigned int imotion = 0; imotion < 6; imotion++) {
          H5Easy::dump(HDF5_file, bodyMotionPath + "/AddedMassXDerivative/DOF_" + std::to_string(imotion),
                       body->GetXDerivativeAddedMass(bodyMotion, imotion));
          auto DOF = AddedMassXDerivativeGroup.getDataSet("DOF_" + std::to_string(imotion));
          DOF.createAttribute<std::string>("Description",
                                           "x-derivative of the added mass coefficients for an acceleration of body " +
                                           std::to_string(bodyMotion->GetID()) +
                                           " and force on body " +
                                           std::to_string(body->GetID()) + ".");
          //TODO : complete units, depending on DOF, bodyMotion, etc.
          DOF.createAttribute<std::string>("Unit", "");
        }

        // Writing the radiation damping matrix for the body.
        auto RadiationDampingXDerivativeGroup = bodyMotionGroup.createGroup("RadiationDampingXDerivative");
        RadiationDampingXDerivativeGroup.createAttribute<std::string>("Description",
                                                                      "x-derivative of the damping coefficients for velocity of body " +
                                                                      std::to_string(bodyMotion->GetID()) +
                                                                      " that radiates waves and generates forces on body " +
                                                                      std::to_string(body->GetID()) + ".");
        for (unsigned int imotion = 0; imotion < 6; imotion++) {
          H5Easy::dump(HDF5_file, bodyMotionPath + "/RadiationDampingXDerivative/DOF_" + std::to_string(imotion),
                       body->GetXDerivativeRadiationDamping(bodyMotion, imotion));
          auto DOF = RadiationDampingXDerivativeGroup.getDataSet("DOF_" + std::to_string(imotion));
          DOF.createAttribute<std::string>("Description",
                                           "x-derivative of the wave damping coefficients for an acceleration of body " +
                                           std::to_string(bodyMotion->GetID()) +
                                           " and force on body " +
                                           std::to_string(body->GetID()) + ".");
          //TODO : complete units, depending on DOF, bodyMotion, etc.
          DOF.createAttribute<std::string>("Unit", "");
        }
      }

      if (body->HasIRF()) {
        // Writing the impulse response function K for the body.
        auto KGroup = bodyMotionGroup.createGroup("ImpulseResponseFunctionK");
        KGroup.createAttribute<std::string>("Description", "Impulse response functions K due to the velocity of body " +
                                                           std::to_string(bodyMotion->GetID()) +
                                                           " that radiates waves and generates forces on body " +
                                                           std::to_string(body->GetID()) + ".");
        for (unsigned int imotion = 0; imotion < 6; imotion++) {
          H5Easy::dump(HDF5_file, bodyMotionPath + "/ImpulseResponseFunctionK/DOF_" + std::to_string(imotion),
                       body->GetIRFInterpolatedData(bodyMotion, "K", imotion, time));
          auto DOF = KGroup.getDataSet("DOF_" + std::to_string(imotion));
          DOF.createAttribute<std::string>("Description", "Impulse response functions K");
        }

        // Writing the impulse response function KU for the body.
        auto KUGroup = bodyMotionGroup.createGroup("ImpulseResponseFunctionKU");
        KUGroup.createAttribute<std::string>("Description",
                                             "Impulse response functions KU due to the velocity of body " +
                                             std::to_string(bodyMotion->GetID()) +
                                             " that radiates waves and generates forces on body " +
                                             std::to_string(body->GetID()) + ".");
        for (unsigned int imotion = 0; imotion < 6; imotion++) {
          H5Easy::dump(HDF5_file, bodyMotionPath + "/ImpulseResponseFunctionKU/DOF_" + std::to_string(imotion),
                       body->GetIRFInterpolatedData(bodyMotion, "KU", imotion, time));
          auto DOF = KUGroup.getDataSet("DOF_" + std::to_string(imotion));
          DOF.createAttribute<std::string>("Description", "Impulse response functions KU");
        }

        // Writing the impulse response function KUXDerivative for the body.
        if (m_hdb->GetIsXDerivative()) {
          auto KUXDerivativeGroup = bodyMotionGroup.createGroup("ImpulseResponseFunctionKUXDerivative");
          KUXDerivativeGroup.createAttribute<std::string>("Description",
                                                          "Impulse response functions KUXDerivative due to the velocity of body " +
                                                          std::to_string(bodyMotion->GetID()) +
                                                          " that radiates waves and generates forces on body " +
                                                          std::to_string(body->GetID()) + ".");
          for (unsigned int imotion = 0; imotion < 6; imotion++) {
            H5Easy::dump(HDF5_file,
                         bodyMotionPath + "/ImpulseResponseFunctionKUXDerivative/DOF_" + std::to_string(imotion),
                         body->GetIRFInterpolatedData(bodyMotion, "KUXDerivative", imotion, time));
            auto DOF = KUXDerivativeGroup.getDataSet("DOF_" + std::to_string(imotion));
            DOF.createAttribute<std::string>("Description", "Impulse response functions KU");
          }
        }

        // Writing the impulse response function KU2 for the body.
        if (m_hdb->GetIsXDerivative()) {
          auto KU2 = bodyMotionGroup.createGroup("ImpulseResponseFunctionKU2");
          KU2.createAttribute<std::string>("Description",
                                           "Impulse response functions KU2 due to the velocity of body " +
                                           std::to_string(bodyMotion->GetID()) +
                                           " that radiates waves and generates forces on body " +
                                           std::to_string(body->GetID()) + ".");
          for (unsigned int imotion = 0; imotion < 6; imotion++) {
            H5Easy::dump(HDF5_file, bodyMotionPath + "/ImpulseResponseFunctionKU2/DOF_" + std::to_string(imotion),
                         body->GetIRFInterpolatedData(bodyMotion, "KU2", imotion, time));
            auto DOF = KU2.getDataSet("DOF_" + std::to_string(imotion));
            DOF.createAttribute<std::string>("Description", "Impulse response functions KU");
          }
        }
      }

      // Writing the modal coefficients
      //TODO: add description, units, etc.
      if (body->HasModal(bodyMotion)) {
        auto modalGroup = bodyMotionGroup.createGroup("Modal");
        modalGroup.createAttribute<std::string>("Description", "Modal coefficients due to the velocity of body " +
                                                               std::to_string(bodyMotion->GetID()) +
                                                               " that radiates waves and generates forces on body " +
                                                               std::to_string(body->GetID()) + ".");
        for (unsigned int idof = 0; idof < 6; idof++) {
          auto DOF = modalGroup.createGroup("DOF_" + std::to_string(idof));

          for (unsigned int iforce = 0; iforce < 6; iforce++) {
            auto force = DOF.createGroup("FORCE_" + std::to_string(iforce));

            auto forcePath = bodyMotionPath + "/Modal/DOF_" + std::to_string(idof) + "/FORCE_" + std::to_string(iforce);
            auto modalCoefficient = body->GetModalCoefficients(bodyMotion, idof, iforce);

            // Real poles and residues.
            auto pole = modalCoefficient.GetRealPoles();
            auto residue = modalCoefficient.GetRealResidues();
            if (pole.size() > 0) {
              H5Easy::dump(HDF5_file, forcePath + "/RealPoles", modalCoefficient.GetRealPoles());
              H5Easy::dump(HDF5_file, forcePath + "/RealResidues", modalCoefficient.GetRealResidues());
            }

            // Complex poles and residues.
            auto pole_cc = modalCoefficient.GetComplexPoles();
            auto residue_cc = modalCoefficient.GetComplexResidues();
            if (pole_cc.cols() > 0) {
              force.createGroup("ComplexPoles");
              force.createGroup("ComplexResidues");
              auto coeff = static_cast<Eigen::VectorXd>(pole_cc.row(0));
              H5Easy::dump(HDF5_file, forcePath + "/ComplexPoles/RealCoeff", coeff);
              coeff = static_cast<Eigen::VectorXd>(pole_cc.row(1));
              H5Easy::dump(HDF5_file, forcePath + "/ComplexPoles/ImagCoeff", coeff);
              coeff = static_cast<Eigen::VectorXd>(residue_cc.row(0));
              H5Easy::dump(HDF5_file, forcePath + "/ComplexResidues/RealCoeff", coeff);
              coeff = static_cast<Eigen::VectorXd>(residue_cc.row(1));
              H5Easy::dump(HDF5_file, forcePath + "/ComplexResidues/ImagCoeff", coeff);
            }

//            auto poleDataSet = force.createDataSet<std::vector<double>>("RealPoles", HighFive::DataSpace::From(pole));
//            poleDataSet.write(pole);
//            auto residueDataSet = force.createDataSet<std::vector<double>>("RealResidues",
//                                                                           HighFive::DataSpace::From(residue));
//            residueDataSet.write(residue);


//            auto complexGroup = force.createGroup("ComplexPoles");
//            auto cplxPoles = modalCoefficient.GetComplexPoles();
//
//            Eigen::VectorXd coeffs = cplxPoles.row(0);
//            complexGroup.createDataSet<Eigen::VectorXd>("RealCoeffs", HighFive::DataSpace::From(coeffs)).write(coeffs);
//            coeffs = cplxPoles.row(1);
//            complexGroup.createDataSet<Eigen::VectorXd>("ImagCoeffs", HighFive::DataSpace::From(coeffs)).write(coeffs);
//
//            complexGroup = force.createGroup("ComplexResidues");
//            auto cplxResidues = modalCoefficient.GetComplexResidues();
//
//            coeffs = cplxResidues.row(0);
//            complexGroup.createDataSet<Eigen::VectorXd>("RealCoeffs", HighFive::DataSpace::From(coeffs)).write(coeffs);
//            coeffs = cplxResidues.row(1);
//            complexGroup.createDataSet<Eigen::VectorXd>("ImagCoeffs", HighFive::DataSpace::From(coeffs)).write(coeffs);

          }
        }
      }

    }

  }

  void HDBWriter::WriteRAO(HighFive::File &HDF5_file, const std::string &path, Body *body) const {

    auto angles = m_hdb->GetWaveDirectionDiscretization();

    for (unsigned int iwaveDir = 0; iwaveDir < angles.size(); ++iwaveDir) {

      auto anglePath = path + "/Angle_" + std::to_string(iwaveDir);

      auto angle = angles(iwaveDir);

      Eigen::MatrixXcd coeff = body->GetRAO(iwaveDir);

      auto angleGroup = HDF5_file.createGroup(anglePath);
      H5Easy::dump(HDF5_file, anglePath + "/Angle", angle * MU_180_PI);
      angleGroup.getDataSet("Angle").createAttribute<std::string>("Description", "Wave direction.");
      angleGroup.getDataSet("Angle").createAttribute<std::string>("Unit", "deg");

      H5Easy::dump(HDF5_file, anglePath + "/Amplitude", static_cast<Eigen::MatrixXd>(coeff.array().abs()));
      angleGroup.getDataSet("Amplitude").createAttribute<std::string>("Description",
                                                                      "Amplitude of the RAO of body " +
                                                                      std::to_string(body->GetID()) +
                                                                      " for a wave direction of " + std::to_string(
                                                                          angle * MU_180_PI) +
                                                                      " deg.");
      angleGroup.getDataSet("Amplitude").createAttribute<std::string>("Unit", "");

      H5Easy::dump(HDF5_file, anglePath + "/Phase",
                   static_cast<Eigen::MatrixXd>(coeff.array().arg() * RAD2DEG)); // Conversion in deg.
      angleGroup.getDataSet("Phase").createAttribute<std::string>("Description",
                                                                  "Phase in deg of the RAO of body " +
                                                                  std::to_string(body->GetID()) +
                                                                  " for a wave direction of " + std::to_string(
                                                                      angle * MU_180_PI) +
                                                                  " deg.");
      angleGroup.getDataSet("Phase").createAttribute<std::string>("Unit", "deg");

//      H5Easy::dump(HDF5_file, anglePath + "/RealCoeffs", static_cast<Eigen::MatrixXd>(coeff.real()));
//      angleGroup.getDataSet("RealCoeffs").createAttribute<std::string>("Description",
//                                                                       "Real part of the Froude-Krylov loads on body " +
//                                                                       std::to_string(body->GetID()) +
//                                                                       " for a wave direction of " + std::to_string(
//                                                                           m_waveDirectionDiscretization.GetVector()[iwaveDir]) +
//                                                                       " deg.");
//      angleGroup.getDataSet("RealCoeffs").createAttribute<std::string>("Unit", "N/m");
//
//      H5Easy::dump(HDF5_file, anglePath + "/ImagCoeffs", static_cast<Eigen::MatrixXd>(coeff.imag()));
//      angleGroup.getDataSet("ImagCoeffs").createAttribute<std::string>("Description",
//                                                                       "Imaginary part of the Froude-Krylov loads on body " +
//                                                                       std::to_string(body->GetID()) +
//                                                                       " for a wave direction of " + std::to_string(
//                                                                           m_waveDirectionDiscretization.GetVector()[iwaveDir]) +
//                                                                       " deg.");
//      angleGroup.getDataSet("ImagCoeffs").createAttribute<std::string>("Unit", "N/m");


    }

  }

  void HDBWriter::WriteWaveDrift(HighFive::File &HDF5_file, const std::string &path, Body *body) const {

    auto waveDriftGroup = HDF5_file.createGroup(path + "/WaveDrift");

    // Write symmetries data
    auto symmetries = body->GetWaveDrift()->GetSymmetries();
    auto sym_x = (int)symmetries[0];
    auto sym_y = (int)symmetries[1];

    HighFive::DataSet dataSet = waveDriftGroup.createDataSet<int>("sym_x", HighFive::DataSpace::From(sym_x));
    dataSet.write(sym_x);
    dataSet.createAttribute<std::string>("Description", "Symmetry along x");

    dataSet = waveDriftGroup.createDataSet<int>("sym_y", HighFive::DataSpace::From(sym_y));
    dataSet.write(sym_y);
    dataSet.createAttribute<std::string>("Description", "Symmetry along y");

    // Write wave drift data
    std::vector<std::string> components = {"surge", "sway", "yaw"};

    auto angles = m_hdb->GetWaveDirectionDiscretization();
    auto frequencies = m_hdb->GetFrequencyDiscretization();
    auto waveDrift = body->GetWaveDrift();

    for (auto &component: components) {

      auto componentGroup = waveDriftGroup.createGroup(component);

      for (unsigned int iangle = 0; iangle < angles.size(); iangle++) {
        auto angle = angles(iangle);
        auto headingGroup = componentGroup.createGroup("angle_" + std::to_string(iangle));
        Eigen::VectorXd data(frequencies.size());
        for (unsigned int i = 0; i < frequencies.size(); i++) {
          data(i) = waveDrift->Eval(component, frequencies(i), angle);
        }
        H5Easy::dump(HDF5_file, path + "/WaveDrift/" + component + "/angle_" + std::to_string(iangle) + "/data", data);
        dataSet = headingGroup.getDataSet("data");
        dataSet.createAttribute<std::string>("Description", "Wave Drift force coefficients");

        dataSet = headingGroup.createDataSet<double>("angle", HighFive::DataSpace::From(angle));
        dataSet.write(angle * MU_180_PI); // Conversion in degrees.
        dataSet.createAttribute<std::string>("Description", "Wave direction angle");
        // TODO : angle unit gestion to add
        dataSet.createAttribute<std::string>("Unit", "rad");

      }
    }
  }

  void HDBWriter::WriteKochin(HighFive::File &HDF5_file) const {
    // Kochin group.
    auto KochinGroup = HDF5_file.createGroup("Kochin");

    // Kochin data.
    double kochin_step = 0.;
    if (m_hdb->GetKochin()) {

      // Kochin angular step.
      kochin_step = m_hdb->GetKochin()->GetKochinStep();

      H5Easy::dump(HDF5_file, "Kochin/KochinStep", kochin_step * MU_180_PI);

      // Wave directions.
      auto angles = m_hdb->GetKochin()->GetWaveDirectionKochin();

      // Diffraction.
      for (unsigned int iwaveDir = 0; iwaveDir < m_hdb->GetKochin()->GetNbKochinDirections(); ++iwaveDir) {

        // Groups.
        auto anglePath = "Kochin/Diffraction/Angle_" + std::to_string(iwaveDir);
        auto angleGroup = HDF5_file.createGroup(anglePath);

        // Wave direction.
        auto angle = angles(iwaveDir); // In rad.
        H5Easy::dump(HDF5_file, anglePath + "/Angle", angle * MU_180_PI); // In deg.
        angleGroup.getDataSet("Angle").createAttribute<std::string>("Description", "Wave direction.");
        angleGroup.getDataSet("Angle").createAttribute<std::string>("Unit", "deg");

        // Kochin function group.
        auto DiffractionKochinGroup = HDF5_file.createGroup(anglePath + "/Function");

        // Kochin function - Real part.
        auto diffraction_kochin = m_hdb->GetKochin()->GetDiffractionKochin(iwaveDir);
        H5Easy::dump(HDF5_file, anglePath + "/Function/RealPart",
                     static_cast<Eigen::MatrixXd>(diffraction_kochin.real()));

        // Kochin function - Imaginary part.
        H5Easy::dump(HDF5_file, anglePath + "/Function/ImagPart",
                     static_cast<Eigen::MatrixXd>(diffraction_kochin.imag()));

        // Derivative Kochin function group.
        auto DiffractionDerivativeKochinGroup = HDF5_file.createGroup(anglePath + "/Derivative");

        // Kochin function derivative.
        if (m_hdb->GetSolver() == "Helios") {
          auto diffraction_kochin_derivative = m_hdb->GetKochin()->GetDiffractionKochinDerivative(iwaveDir);
          H5Easy::dump(HDF5_file, anglePath + "/Derivative/RealPart",
                       static_cast<Eigen::MatrixXd>(diffraction_kochin_derivative.real()));

          // Kochin function derivative.
          H5Easy::dump(HDF5_file, anglePath + "/Derivative/ImagPart",
                       static_cast<Eigen::MatrixXd>(diffraction_kochin_derivative.imag()));
        }
      }

      // Radiation.
      for (unsigned int ibody = 0; ibody < m_hdb->GetNbBodies(); ++ibody) {

        // Body.
        auto body = m_hdb->GetBody(ibody);

        // Dof.
        for (unsigned int idof = 0; idof < 6; idof++) {

          // Group.
          auto dofPath = "Kochin/Radiation/Body_" + std::to_string(ibody) + "/DOF_" + std::to_string(idof);
          auto dofGroup = HDF5_file.createGroup(dofPath);

          // Kochin function group.
          auto RadiationKochinGroup = HDF5_file.createGroup(dofPath + "/Function");

          // Kochin function - Real part.
          auto radiation_kochin = m_hdb->GetKochin()->GetRadiationKochin(body, idof);
          H5Easy::dump(HDF5_file, dofPath + "/Function/RealPart",
                       static_cast<Eigen::MatrixXd>(radiation_kochin.real()));

          // Kochin function - Imaginary part.
          H5Easy::dump(HDF5_file, dofPath + "/Function/ImagPart",
                       static_cast<Eigen::MatrixXd>(radiation_kochin.imag()));

          // Kochin function group.
          if (m_hdb->GetSolver() == "Helios") {
            auto RadiationKochinDerivativeGroup = HDF5_file.createGroup(dofPath + "/Derivative");

            // Kochin function derivative - Real part.
            auto radiation_kochin_derivative = m_hdb->GetKochin()->GetRadiationKochinDerivative(body, idof);
            H5Easy::dump(HDF5_file, dofPath + "/Derivative/RealPart",
                         static_cast<Eigen::MatrixXd>(radiation_kochin_derivative.real()));

            // Kochin function derivative - Imaginary part.
            H5Easy::dump(HDF5_file, dofPath + "/Derivative/ImagPart",
                         static_cast<Eigen::MatrixXd>(radiation_kochin_derivative.imag()));
          }
        }

      }

    }
    /*else {
      // Kochin angular step.
      kochin_step = m_hdb->GetWaveDrift()->GetKochinStep();
    }
    dataSet = KochinGroup.createDataSet<double>("KochinStep", HighFive::DataSpace::From(kochin_step));
    dataSet.write(kochin_step * MU_180_PI); // Conversion in degrees.
    dataSet.createAttribute<std::string>("Description", "Angular discretization in degrees for the Kochin functions");
  */
  }

  void HDBWriter::WriteWaveField(HighFive::File &file) const {

    auto VF = file.createGroup("WaveField");

  }

  void HDBWriter::WriteVF(HighFive::File &file) const {

    auto VF = file.createGroup("VectorFitting");
    H5Easy::dump(file, "VectorFitting/Relaxed", static_cast<int> (m_hdb->GetVFRelaxed()));
    H5Easy::dump(file, "VectorFitting/MaxOrder", static_cast<int> (m_hdb->GetVFMaxOrder()));
    H5Easy::dump(file, "VectorFitting/Tolerance", static_cast<double> (m_hdb->GetVFTolerance()));

  }

  void HDBWriter::WriteExpertParameters(HighFive::File &file) const {

    // This method writes the expert numerical parameters.

    auto ExpertParameters = file.createGroup("ExpertParameters");

    H5Easy::dump(file, "ExpertParameters/SurfaceIntegrationOrder",
                 static_cast<int> (m_hdb->GetSurfaceIntegrationOrder()));

    auto green_function = m_hdb->GetGreenFunction();
    HighFive::DataSet dataSet = file.createDataSet<std::string>("ExpertParameters/GreenFunction",
                                                                HighFive::DataSpace::From(green_function));
    dataSet.write(green_function);
    dataSet.createAttribute<std::string>("Description", "Green function.");

    H5Easy::dump(file, "ExpertParameters/Crmax", static_cast<int> (m_hdb->GetCrmax()));

  }

  void export_HDB(const std::string &filename, HydrodynamicDataBase *hdb) {

    auto hdbWriter = std::make_shared<HDBWriter>(hdb);
    hdbWriter->Write(filename);

  }

}
