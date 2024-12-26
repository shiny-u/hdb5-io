//
// Created by lletourn on 14/04/20.
//

#include "HDBReader.h"

#include "hdb5_io/containers/HydrodynamicDataBase.h"
#include "hdb5_io/containers/Body.h"
#include "hdb5_io/containers/WaveDrift.h"
#include "hdb5_io/containers/Kochin.h"

#include <highfive/H5File.hpp>
#include <highfive/H5Easy.hpp>

#include <filesystem>

namespace hdb5_io {

  void HDBReader::Read(const std::string &filename) {

    // HDB file.
    HighFive::File file(filename, HighFive::File::ReadOnly);

    // HDB basic data (version, date, gravity constant, density, etc.).
    ReadHDBBasics(file);

    // Wave frequency and wave directions.
    ReadDiscretizations(file);

    // Symmetries.
    if (file.exist("Symmetries")) {
      ReadSymmetries(file);
    }

    // Body basic data (index, name, position, mass, etc.).
    std::vector<Body *> bodies;
    for (int i = 0; i < m_hdb->GetNbBodies(); i++) {
      bodies.push_back(ReadBodyBasics(file, "Bodies/Body_" + std::to_string(i)));
    }

    // Other body data.
    for (auto &body : bodies) {

      // Body mesh.
      if (file.getGroup("Bodies/Body_" + std::to_string(body->GetID())).exist("Mesh")) {
        ReadMesh(file, "Bodies/Body_" + std::to_string(body->GetID()) + "/Mesh", body);
      }

      // Diffraction loads.
      ReadExcitation(Diffraction, file, "Bodies/Body_" + std::to_string(body->GetID()) + "/Excitation/Diffraction", body);

      // x-derivative of the diffraction loads.
      if (file.getGroup("Bodies/Body_" + std::to_string(body->GetID()) + "/Excitation").exist("DiffractionXDerivative")) {
        m_hdb->IsXDerivative();
        ReadExcitation(DiffractionXDerivative, file, "Bodies/Body_" + std::to_string(body->GetID()) + "/Excitation/DiffractionXDerivative", body);
      }

      // Froude-Krylov loads.
      ReadExcitation(Froude_Krylov, file, "Bodies/Body_" + std::to_string(body->GetID()) + "/Excitation/FroudeKrylov", body);

      // x-derivative of the diffraction loads.
      if (m_hdb->GetIsXDerivative()) {
        ReadExcitation(Froude_KrylovXDerivative, file, "Bodies/Body_" + std::to_string(body->GetID()) + "/Excitation/FroudeKrylovXDerivative", body);
      }

      // Excitation loads.
      body->ComputeExcitation();

      // x-derivative of the excitation loads.
      if (m_hdb->GetIsXDerivative()) {
        body->ComputeXDerivativeExcitation();
      }

      // Added mass, damping, IRF, poles and residues and the x-derivatives.
      ReadRadiation(file, "Bodies/Body_" + std::to_string(body->GetID()) + "/Radiation", body);

      // RAOs.
      if (file.getGroup("Bodies/Body_" + std::to_string(body->GetID())).exist("RAO")) {
        ReadRAO(file, "Bodies/Body_" + std::to_string(body->GetID()) + "/RAO", body);
      }
    }

    // Mean wave drift loads.
    ReadWaveDrift(file);

    // Wave field parameters.
    if (file.exist("WaveField")) {
      ReadWaveField(file);
    }

    // Vector fitting parameters.
    if (file.exist("VectorFitting")) {
      ReadVectorFitting(file);
    }

    // Expert numerical paramaters.
    if (file.exist("ExpertParameters")) {
      ReadExpertNumericalParameters(file);
    }

  }

  void HDBReader::ReadHDBBasics(const HighFive::File &HDF5_file) {

    m_hdb->SetCreationDate(H5Easy::load<std::string>(HDF5_file, "CreationDate"));
    m_hdb->SetSolver(H5Easy::load<std::string>(HDF5_file, "Solver"));
    if(m_hdb->GetSolver() == "Helios") {
      if (HDF5_file.exist("NormalizedCommitHash")) {
        m_hdb->SetNormalizedVersionString(H5Easy::load<std::string>(HDF5_file, "NormalizedCommitHash"));
      }
    }
    m_hdb->SetNormalizationLength(H5Easy::load<double>(HDF5_file, "NormalizationLength"));
    m_hdb->SetGravityAcceleration(H5Easy::load<double>(HDF5_file, "GravityAcc"));
    m_hdb->SetWaterDensity(H5Easy::load<double>(HDF5_file, "WaterDensity"));
    m_hdb->SetWaterDepth(H5Easy::load<double>(HDF5_file, "WaterDepth"));
    m_hdb->SetNbBodies(H5Easy::load<int>(HDF5_file, "NbBody"));

  }

  void
  HDBReader::ReadExcitation(HDBReader::excitationType type, const HighFive::File &HDF5_file, const std::string &path,
                            Body *body) {
    auto forceMask = body->GetForceMask();

    for (unsigned int iwaveDir = 0; iwaveDir < m_hdb->GetWaveDirectionDiscretization().size(); ++iwaveDir) {

      auto WaveDirPath = path + "/Angle_" + std::to_string(iwaveDir);

//      auto angle = H5Easy::load<double>(HDF5_file, WaveDirPath + "/Angle");
//      assert(abs(m_waveDirectionDiscretization.GetVector()[iwaveDir] - angle) < 1E-5);

      auto realCoeffs = H5Easy::load<Eigen::MatrixXd>(HDF5_file, WaveDirPath + "/RealCoeffs");
      auto imagCoeffs = H5Easy::load<Eigen::MatrixXd>(HDF5_file, WaveDirPath + "/ImagCoeffs");
      auto Dcoeffs = realCoeffs + MU_JJ * imagCoeffs;

      Eigen::MatrixXcd ExcitationCoeffs;
      if (imagCoeffs.rows() != 6) {
        assert(imagCoeffs.rows() == forceMask.GetNbDOF());
        ExcitationCoeffs.setZero();
        for (int i = 0; i < forceMask.GetNbDOF(); i++) {
          ExcitationCoeffs.row(forceMask.GetListDOF()[i]) = Dcoeffs.row(i);
        }
//        // Condense the matrix by removing the lines corresponding to the masked DOFs
//        ExcitationCoeffs = Eigen::VectorXi::Map(forceMask.GetListDOF().data(), forceMask.GetNbDOF()).replicate(1,Dcoeffs.cols()).unaryExpr(Dcoeffs);
      } else {
        ExcitationCoeffs = Dcoeffs;
      }

      switch (type) {
        case Diffraction : {
          body->SetDiffraction(iwaveDir, ExcitationCoeffs);
          break;
        }
        case DiffractionXDerivative : {
          body->SetXDerivativeDiffraction(iwaveDir, ExcitationCoeffs);
          break;
        }
        case Froude_Krylov : {
          body->SetFroudeKrylov(iwaveDir, ExcitationCoeffs);
          break;
        }
        case Froude_KrylovXDerivative : {
          body->SetXDerivativeFroudeKrylov(iwaveDir, ExcitationCoeffs);
          break;
        }
      }

    }

  }

  void HDBReader::ReadRAO(const HighFive::File &HDF5_file, const std::string &path, Body *body) {
    auto forceMask = body->GetForceMask();

    for (unsigned int iwaveDir = 0; iwaveDir < m_hdb->GetWaveDirectionDiscretization().size(); ++iwaveDir) {

      auto WaveDirPath = path + "/Angle_" + std::to_string(iwaveDir);

//      auto angle = H5Easy::load<double>(HDF5_file, WaveDirPath + "/Angle");
//      assert(abs(m_waveDirectionDiscretization.GetVector()[iwaveDir] - angle) < 1E-5);

      auto amplitude = H5Easy::load<Eigen::MatrixXd>(HDF5_file, WaveDirPath + "/Amplitude");
      auto phase = H5Easy::load<Eigen::MatrixXd>(HDF5_file, WaveDirPath + "/Phase");

      auto DataSet = HDF5_file.getDataSet(WaveDirPath + "/Phase");
      std::string unit;
      DataSet.getAttribute("Unit").read<std::string>(unit);

      // Converstion in rad.
      if (unit != "rad")
        phase = phase.array() * DEG2RAD;

      Eigen::MatrixXcd Dcoeffs = amplitude.array() * Eigen::exp(MU_JJ * phase.array());

      Eigen::MatrixXcd raoCoeffs;
      if (amplitude.rows() != 6) {
        assert(amplitude.rows() == forceMask.GetNbDOF());
        raoCoeffs.setZero();
        for (int i = 0; i < forceMask.GetNbDOF(); i++) {
          raoCoeffs.row(forceMask.GetListDOF()[i]) = Dcoeffs.row(i);
        }
//        // Condense the matrix by removing the lines corresponding to the masked DOFs
//        raoCoeffs = Eigen::VectorXi::Map(forceMask.GetListDOF().data(), forceMask.GetNbDOF()).replicate(1,Dcoeffs.cols()).unaryExpr(Dcoeffs);
      } else {
        raoCoeffs = Dcoeffs;
      }

      body->SetRAO(iwaveDir, raoCoeffs);

    }


  }

  std::vector<Eigen::MatrixXd>
  HDBReader::ReadComponents(const HighFive::File &HDF5_file, const std::string &path,
                            Eigen::Matrix<bool, 6, 6> radiationMask) {

    std::vector<Eigen::MatrixXd> impulseResponseFunctionsK;

    Mask motionMask;
    Eigen::MatrixXd IRFCoeffs;

    for (unsigned int imotion = 0; imotion < 6; ++imotion) {
      auto IRF = H5Easy::load<Eigen::MatrixXd>(HDF5_file, path + "/DOF_" + std::to_string(imotion));
      motionMask.SetMask(radiationMask.row(imotion));
      if (IRF.rows() != 6) {
        assert(IRF.rows() == motionMask.GetNbDOF());
        IRFCoeffs.setZero();
        for (int i = 0; i < motionMask.GetNbDOF(); i++) {
          IRFCoeffs.row(motionMask.GetListDOF()[i]) = IRF.row(i);
        }
//          // Condense the matrix by removing the lines corresponding to the masked DOFs
//          //TODO:: passer en fonction de MathUtils ?
//          IRFCoeffs = Eigen::VectorXi::Map(motionMask.GetListDOF().data(), motionMask.GetNbDOF()).replicate(1,IRF.cols()).unaryExpr(IRF);
      } else {
        IRFCoeffs = IRF;
      }
      impulseResponseFunctionsK.push_back(IRFCoeffs);
    }

    return impulseResponseFunctionsK;

  }

  void HDBReader::ReadMesh(HighFive::File &HDF5_file, const std::string &path, Body *body) {

    auto nbVertices = H5Easy::load<int>(HDF5_file, path + "/NbVertices");
    auto nbFaces = H5Easy::load<int>(HDF5_file, path + "/NbFaces");

    if (nbFaces ==0 && nbVertices==0) return;

    auto vertices_hdb = H5Easy::load<Eigen::MatrixXd>(HDF5_file, path + "/Vertices");
    auto faces_hdb = H5Easy::load<Eigen::MatrixXi>(HDF5_file, path + "/Faces");

    std::vector<mathutils::Vector3d<double>> vertices;
    std::vector<Eigen::VectorXi> faces;

    for (unsigned int i = 0; i < nbVertices; i++) {
      mathutils::Vector3d<double> vertex = vertices_hdb.row(i);
      vertices.emplace_back(vertex);
    }

    for (unsigned int i = 0; i < nbFaces; i++) {
      Eigen::VectorXi face = faces_hdb.row(i);
      faces.emplace_back(face);
    }

    #ifdef MESH_SUPPORT
    body->LoadMesh(vertices, faces);
    #endif

  }

  void HDBReader::ReadWaveField(HighFive::File &file) {

    m_hdb->SetWaveField();

  }

  void HDBReader::ReadVectorFitting(HighFive::File &file) {

    m_hdb->SetVF();
    m_hdb->SetVFRelaxed(H5Easy::load<int>(file, "VectorFitting/Relaxed"));
    m_hdb->SetVFMaxOrder(H5Easy::load<int>(file, "VectorFitting/MaxOrder"));
    m_hdb->SetVFTolerance(H5Easy::load<double>(file, "VectorFitting/Tolerance"));

  }

  void HDBReader::ReadSymmetries(HighFive::File &file) {

    m_hdb->SetSymmetries();
    m_hdb->SetSymBottom(H5Easy::load<int>(file, "Symmetries/Bottom"));
    m_hdb->SetSymXOZ(H5Easy::load<int>(file, "Symmetries/xOz"));
    m_hdb->SetSymYOZ(H5Easy::load<int>(file, "Symmetries/yOz"));

  }

  void HDBReader::ReadExpertNumericalParameters(HighFive::File &file) {

    // This method reads the expert numerical parameters.

    m_hdb->SetExpertParameters();
    m_hdb->SetSurfaceIntegrationOrder(H5Easy::load<int>(file, "ExpertParameters/SurfaceIntegrationOrder"));
    m_hdb->SetGreenFunction(H5Easy::load<std::string>(file, "ExpertParameters/GreenFunction"));
    m_hdb->SetCrmax(H5Easy::load<int>(file, "ExpertParameters/Crmax"));

  }

  std::shared_ptr<HydrodynamicDataBase> import_HDB(const std::string &filename) {

    if(!std::filesystem::is_regular_file(filename)) {
      std::cerr << "File " << filename << " not FOUND." << std::endl;
      exit(EXIT_FAILURE);
    }

    auto hdb = std::make_shared<HydrodynamicDataBase>();

    HighFive::File file(filename, HighFive::File::ReadOnly);

    double version = 1.0;
    if (file.exist("Version"))
      version = H5Easy::load<double>(file, "Version");

    if (!mathutils::IsClose(version, HDB_NUM_VERSION)) {
      std::cerr << "Num version v" << version << " of file " << filename << " different from request version " << HDB_NUM_VERSION << std::endl;
      exit(EXIT_FAILURE);
    }

    auto hdb_reader = std::make_shared<HDBReader>(hdb.get());
    hdb_reader->Read(filename);

    return hdb;
  }

  Eigen::VectorXd
  HDBReader::ReadWaveDriftComponents(HighFive::File &HDF5_file, const std::string &path, unsigned int i) {
    return H5Easy::load<Eigen::VectorXd>(HDF5_file, path + "/angle_" + std::to_string(i) + "/data");
  }

  void HDBReader::ReadWaveDrift(HighFive::File &HDF5_file) {

    for (int i_body=0; i_body<m_hdb->GetNbBodies(); i_body++) {

      auto body = m_hdb->GetBody(i_body);
      auto body_path = "Bodies/Body_" + std::to_string(body->GetID());

      if (HDF5_file.exist(body_path+"/WaveDrift")) {

        auto waveDrift = std::make_shared<WaveDrift>();

        auto frequency = m_hdb->GetFrequencyDiscretization();
        auto waveDirection = m_hdb->GetWaveDirectionDiscretization();

        waveDrift->SetFrequencies(frequency);
        waveDrift->SetWaveDirections(waveDirection);

        auto sym_X = H5Easy::load<int>(HDF5_file, body_path + "/WaveDrift/sym_x");
        auto sym_Y = H5Easy::load<int>(HDF5_file, body_path + "/WaveDrift/sym_y");

        waveDrift->SetSymmetries(sym_X == 1, sym_Y == 1);

        Eigen::MatrixXd surge(waveDirection.size(), frequency.size());
        Eigen::MatrixXd sway(waveDirection.size(), frequency.size());
        Eigen::MatrixXd yaw(waveDirection.size(), frequency.size());

        for (unsigned int i = 0; i < waveDirection.size(); i++) {
          auto data_surge = ReadWaveDriftComponents(HDF5_file, body_path + "/WaveDrift/surge", i);
          surge.row(i) = data_surge;
          auto data_sway = ReadWaveDriftComponents(HDF5_file, body_path + "/WaveDrift/sway", i);
          sway.row(i) = data_sway;
          auto data_yaw = ReadWaveDriftComponents(HDF5_file, body_path + "/WaveDrift/yaw", i);
          yaw.row(i) = data_yaw;
        }

        std::vector<double> coeff_surge(&surge(0, 0), surge.data() + surge.size());
        waveDrift->AddData("surge", coeff_surge);
        std::vector<double> coeff_sway(&sway(0, 0), sway.data() + sway.size());
        waveDrift->AddData("sway", coeff_sway);
        std::vector<double> coeff_yaw(&yaw(0, 0), yaw.data() + yaw.size());
        waveDrift->AddData("yaw", coeff_yaw);

        body->SetWaveDrift(waveDrift);
      }
    }
    ReadKochin(HDF5_file);
  }

  void HDBReader::ReadDiscretizations(const HighFive::File &file) {

    m_hdb->SetFrequencyDiscretization(H5Easy::load<Eigen::VectorXd>(file, "Discretizations/Frequency"));
    if (file.exist("Discretizations/Time")) {
      m_hdb->SetTimeDiscretization(H5Easy::load<Eigen::VectorXd>(file, "Discretizations/Time"));
    }

    // The wave directions are read in degrees from the hdb5 file. The conversion degrees to radians is performed below.
    m_hdb->SetWaveDirectionDiscretization(
        H5Easy::load<Eigen::VectorXd>(file, "Discretizations/WaveDirection") * MU_PI_180);

  }

  void HDBReader::ReadRadiation(const HighFive::File &file, const std::string &path, Body *body) {

    for (int ibodyMotion = 0; ibodyMotion < m_hdb->GetNbBodies(); ++ibodyMotion) {

      auto bodyMotion = m_hdb->GetBody(ibodyMotion);
      auto bodyMotionPath = path + "/BodyMotion_" + std::to_string(ibodyMotion);

      // Reading the infinite added mass matrix for the body.

      if (file.exist( bodyMotionPath + "/InfiniteAddedMass")) {
        auto infiniteAddedMass = H5Easy::load<Eigen::MatrixXd>(file, bodyMotionPath + "/InfiniteAddedMass");
        body->SetInfiniteAddedMass(bodyMotion, infiniteAddedMass);
      }

      if (m_hdb->GetIsXDerivative()) {
        auto infiniteAddedMassXDerivative = H5Easy::load<Eigen::MatrixXd>(file, bodyMotionPath + "/InfiniteAddedMassXDerivative");
        body->SetXDerivativeInfiniteAddedMass(bodyMotion, infiniteAddedMassXDerivative);
      }

      // Reading the radiation mask matrix for the body.
      auto radiationMask = H5Easy::load<Eigen::Matrix<int, 6, 6>>(file, bodyMotionPath + "/RadiationMask");
      auto mask = radiationMask.cast<bool>();
      body->SetRadiationMask(bodyMotion, mask);

      // Reading the impulse response functions.
      if (file.exist(bodyMotionPath + "/ImpulseResponseFunctionK")) {
        auto impulseResponseFunctionsK = ReadComponents(file, bodyMotionPath + "/ImpulseResponseFunctionK", mask);
        body->SetIRF(bodyMotion, "K", impulseResponseFunctionsK);
      }

      if (file.exist(bodyMotionPath + "/ImpulseResponseFunctionKU")) {
        auto impulseResponseFunctionsK = ReadComponents(file, bodyMotionPath + "/ImpulseResponseFunctionKU", mask);
        body->SetIRF(bodyMotion, "KU", impulseResponseFunctionsK);
      }

      if (m_hdb->GetIsXDerivative()) {
        if (file.exist(bodyMotionPath + "/ImpulseResponseFunctionKUXDerivative")) {
          auto impulseResponseFunctionsK = ReadComponents(file, bodyMotionPath + "/ImpulseResponseFunctionKUXDerivative", mask);
          body->SetIRF(bodyMotion, "KUXDerivative", impulseResponseFunctionsK);
        }
      }

      if (m_hdb->GetIsXDerivative()) {
        if (file.exist(bodyMotionPath + "/ImpulseResponseFunctionKU2")) {
          auto impulseResponseFunctionsK = ReadComponents(file, bodyMotionPath + "/ImpulseResponseFunctionKU2", mask);
          body->SetIRF(bodyMotion, "KU2", impulseResponseFunctionsK);
        }
      }

      // Reading the added mass and radiation damping coefficients
      auto addedMass = ReadComponents(file, bodyMotionPath + "/AddedMass", mask);
      std::vector<mathutils::Matrix66<double>> AM_tmp;
      AM_tmp.reserve(m_hdb->GetFrequencyDiscretization().size());
      for (int iw=0; iw < m_hdb->GetFrequencyDiscretization().size(); ++iw) {
        mathutils::Matrix66<double> tmp_matrix;
        for (int imotion = 0; imotion < 6; ++imotion) {
          for (int iforce = 0; iforce < 6; ++iforce) {
            tmp_matrix(iforce,imotion) = addedMass[imotion](iforce,iw);
          }
        }
        AM_tmp.push_back(tmp_matrix);
      }
      body->SetAddedMass(bodyMotion, AM_tmp);

      auto radiationDamping = ReadComponents(file, bodyMotionPath + "/RadiationDamping", mask);
      std::vector<mathutils::Matrix66<double>> RD_tmp;
      RD_tmp.reserve(m_hdb->GetFrequencyDiscretization().size());
      for (int iw=0; iw < m_hdb->GetFrequencyDiscretization().size(); ++iw) {
        mathutils::Matrix66<double> tmp_matrix;
        for (int imotion = 0; imotion < 6; ++imotion) {
          for (int iforce = 0; iforce < 6; ++iforce) {
            tmp_matrix(iforce,imotion) = radiationDamping[imotion](iforce,iw);
          }
        }
        RD_tmp.push_back(tmp_matrix);
      }
      body->SetRadiationDamping(bodyMotion, RD_tmp);

      // x-derivative of the radiation coefficients.
      if (m_hdb->GetIsXDerivative()) {
        auto addedMassXDerivative = ReadComponents(file, bodyMotionPath + "/AddedMassXDerivative", mask);
        std::vector<mathutils::Matrix66<double>> AMXDerivative_tmp;
        AMXDerivative_tmp.reserve(m_hdb->GetFrequencyDiscretization().size());
        for (int iw=0; iw < m_hdb->GetFrequencyDiscretization().size(); ++iw) {
          mathutils::Matrix66<double> tmp_matrix;
          for (int imotion = 0; imotion < 6; ++imotion) {
            for (int iforce = 0; iforce < 6; ++iforce) {
              tmp_matrix(iforce,imotion) = addedMassXDerivative[imotion](iforce,iw);
            }
          }
          AMXDerivative_tmp.push_back(tmp_matrix);
        }
        body->SetXDerivativeAddedMass(bodyMotion, AMXDerivative_tmp);

        auto radiationDampingXDerivative = ReadComponents(file, bodyMotionPath + "/RadiationDampingXDerivative", mask);
        std::vector<mathutils::Matrix66<double>> RDXDerivative_tmp;
        RDXDerivative_tmp.reserve(m_hdb->GetFrequencyDiscretization().size());
        for (int iw=0; iw < m_hdb->GetFrequencyDiscretization().size(); ++iw) {
          mathutils::Matrix66<double> tmp_matrix;
          for (int imotion = 0; imotion < 6; ++imotion) {
            for (int iforce = 0; iforce < 6; ++iforce) {
              tmp_matrix(iforce,imotion) = radiationDampingXDerivative[imotion](iforce,iw);
            }
          }
          RDXDerivative_tmp.push_back(tmp_matrix);
        }
        body->SetXDerivativeRadiationDamping(bodyMotion, RDXDerivative_tmp);
      }
    }

    for (int ibodyMotion = 0; ibodyMotion < m_hdb->GetNbBodies(); ++ibodyMotion) {

      auto bodyMotion = m_hdb->GetBody(ibodyMotion);
      auto bodyMotionPath = path + "/BodyMotion_" + std::to_string(ibodyMotion);

      if (file.exist(bodyMotionPath + "/ZeroFreqAddedMass")) {
        auto zeroFreqAddedMass = H5Easy::load<Eigen::MatrixXd>(file, bodyMotionPath + "/ZeroFreqAddedMass");
        body->SetZeroFreqAddedMass(bodyMotion, zeroFreqAddedMass);
      }

      if (m_hdb->GetIsXDerivative()) {
        if (file.exist(bodyMotionPath + "/ZeroFreqAddedMassXDerivative")) {
          auto zeroFreqAddedMassXDerivative = H5Easy::load<Eigen::MatrixXd>(file, bodyMotionPath + "/ZeroFreqAddedMassXDerivative");
          body->SetXDerivativeZeroFreqAddedMass(bodyMotion, zeroFreqAddedMassXDerivative);
        }
      }

      if (file.exist(bodyMotionPath + "/Modal")) {

        for (unsigned int idof = 0; idof < 6; idof++) {

          std::vector<PoleResidue> modalCoeff;
          for (unsigned int iforce = 0; iforce < 6; iforce++) {

            auto forcePath = bodyMotionPath + "/Modal/DOF_" + std::to_string(idof) + "/FORCE_" + std::to_string(iforce);

            // Real poles and residues.
            int nPoles_real = 0;
            Eigen::VectorXd poles, residues;
            if (file.exist(forcePath + "/RealPoles")) {
              poles = H5Easy::load<Eigen::VectorXd>(file, forcePath + "/RealPoles");
              nPoles_real = poles.size();
              residues = H5Easy::load<Eigen::VectorXd>(file, forcePath + "/RealResidues");
              assert(residues.size() == nPoles_real);
            }

            // Complex poles and residues.
            int nPoles_cc = 0;
            Eigen::VectorXcd cplxPoles, cplxResidues;
            if (file.exist(forcePath + "/ComplexPoles/RealCoeff")) {
              // Poles.
              auto realCoeff = H5Easy::load<Eigen::VectorXd>(file, forcePath + "/ComplexPoles/RealCoeff");
              auto imagCoeff = H5Easy::load<Eigen::VectorXd>(file, forcePath + "/ComplexPoles/ImagCoeff");
              nPoles_cc = realCoeff.size();
              assert(imagCoeff.size() == nPoles_cc);
              cplxPoles = realCoeff + MU_JJ * imagCoeff;

              // Residues.
              realCoeff = H5Easy::load<Eigen::VectorXd>(file, forcePath + "/ComplexResidues/RealCoeff");
              imagCoeff = H5Easy::load<Eigen::VectorXd>(file, forcePath + "/ComplexResidues/ImagCoeff");
              assert(realCoeff.size() == nPoles_cc && imagCoeff.size() == nPoles_cc);
              cplxResidues = realCoeff + MU_JJ * imagCoeff;
            }

            // Adding to modalCoeff.
            PoleResidue pair;
            for (int i = 0; i < nPoles_real; i++) {
              pair.AddPoleResidue(poles(i), residues(i));
            }
            for (int i = 0; i < nPoles_cc; i++) {
              pair.AddPoleResidue(cplxPoles(i), cplxResidues(i));
            }
            modalCoeff.emplace_back(pair);

          }
          body->AddModalCoefficients(bodyMotion, modalCoeff);

        }

      }

    }

  }

  Body *HDBReader::ReadBodyBasics(const HighFive::File &file, const std::string &path) {

    // Name.
//    std::string name;
    auto name = H5Easy::load<std::string>(file, path + "/BodyName");

    // Index.
    auto id = H5Easy::load<unsigned int>(file, path + "/ID");

    // New body.
    auto body = m_hdb->NewBody(id, name);

    // Horizontal position in world.
    if (file.exist(path + "/BodyPosition")) {
      auto BodyPosition = H5Easy::load<Eigen::Vector3d>(file, path + "/BodyPosition");
      Eigen::Vector3d HorizontalPosition = Eigen::Vector3d::Zero();
      HorizontalPosition(0) = BodyPosition(0);
      HorizontalPosition(1) = BodyPosition(1);
      HorizontalPosition(2) = 0.;
      body->SetHorizontalPositionInWorld(HorizontalPosition);
    } else if(file.exist(path + "/HorizontalPosition/x") && file.exist(path + "/HorizontalPosition/y") &&
              file.exist(path + "/HorizontalPosition/psi")) {
      auto x = H5Easy::load<double>(file, path + "/HorizontalPosition/x");
      auto y = H5Easy::load<double>(file, path + "/HorizontalPosition/y");
      auto psi = H5Easy::load<double>(file, path + "/HorizontalPosition/psi"); // Degrees.
      Eigen::Vector3d HorizontalPosition = Eigen::Vector3d::Zero();
      HorizontalPosition(0) = x;
      HorizontalPosition(1) = y;
      HorizontalPosition(2) = psi;
      body->SetHorizontalPositionInWorld(HorizontalPosition);
    }

    // Computation point in body frame.
    if (file.exist(path + "/ComputationPoint")) {
      body->SetComputationPointInBodyFrame(H5Easy::load<Eigen::Vector3d>(file, path + "/ComputationPoint"));
    } else {
      body->SetComputationPointInBodyFrame(Eigen::Vector3d::Zero());
    }

    if (file.exist(path + "/WaveReferencePoint")) {
      body->SetWaveReferencePointInBodyFrame(H5Easy::load<Eigen::Vector2d>(file, path + "/WaveReferencePoint"));
    } else {
      body->SetWaveReferencePointInBodyFrame(mathutils::Vector2d<double>::Zero());
    }

    // Hydrostatic matrix.
    if (file.exist(path + "/Hydrostatic")) {
      mathutils::Matrix66<double> stiffnessMatrix;
      stiffnessMatrix = H5Easy::load<Eigen::Matrix<double, 6, 6>>(file, path + "/Hydrostatic/StiffnessMatrix");
      body->SetStiffnessMatrix(stiffnessMatrix);
    }

    // Inertia matrix.
    if (file.exist(path + "/Inertia")) {
      mathutils::Matrix66<double> inertiaMatrix;
      inertiaMatrix = H5Easy::load<Eigen::Matrix<double, 6, 6>>(file, path + "/Inertia/InertiaMatrix");
      body->SetInertia(inertiaMatrix);
    }

    // Mooring matrix.
    if (file.exist(path + "/Mooring")) {
      mathutils::Matrix66<double> mooringMatrix;
      mooringMatrix = H5Easy::load<Eigen::Matrix<double, 6, 6>>(file, path + "/Mooring/MooringMatrix");
      body->SetMooring(mooringMatrix);
    }

    // Damping matrix.
    if (file.exist(path + "/LinearDamping")) {
      mathutils::Matrix66<double> dampingMatrix;
      dampingMatrix = H5Easy::load<Eigen::Matrix<double, 6, 6>>(file, path + "/LinearDamping/DampingMatrix");
      body->SetLinearDamping(dampingMatrix);
    }

    // Force and Motion masks.
    //TODO : move Mask to Excitation folder once it has be done in HDB5Tool too
    body->SetForceMask(H5Easy::load<Eigen::Matrix<bool, 6, 1>>(file, path + "/Mask/ForceMask"));

    return body;

  }

  void HDBReader::ReadKochin(const HighFive::File& HDF5_file) {

    // Kochin functions.
    if (HDF5_file.exist("Kochin")) {

      // Kochin angular step.
      double kochin_step;
      if (HDF5_file.exist("Kochin/KochinStep")) {
        kochin_step = H5Easy::load<double>(HDF5_file, "Kochin/KochinStep"); // In degrees.
      } else {
        kochin_step = H5Easy::load<double>(HDF5_file, "KochinStep"); // In degrees.
      }

      if (HDF5_file.exist("Kochin/Diffraction") && HDF5_file.exist("Kochin/Radiation")) {

        // Number of wave directions for the Kochin functions (may be different from the wave directions of the hdb
        // in case of symmetry).
        auto nbDirKochin = HDF5_file.getGroup("Kochin/Diffraction/").getNumberObjects();
        auto wave_direction_kochin = mathutils::VectorN<double>(nbDirKochin); // In rad.

        auto kochin = std::make_shared<Kochin>(m_hdb, kochin_step * MU_PI_180, nbDirKochin); // Conversion in radians.

        // Diffraction Kochin functions and their derivatives.
        auto root = "Kochin/Diffraction/";
        for (unsigned int iwaveDir = 0; iwaveDir < nbDirKochin; ++iwaveDir) {

          auto obj = "Angle_" + std::to_string(iwaveDir);

          // Wave direction for Kochin functions.
          auto angle = H5Easy::load<double>(HDF5_file, root + obj + "/Angle");
          wave_direction_kochin(iwaveDir) = angle * MU_PI_180; // In rad.

          auto diffraction_kochin_real_part = H5Easy::load<Eigen::MatrixXd>(HDF5_file,
                                                                            root + obj + "/Function/RealPart");
          auto diffraction_kochin_imag_part = H5Easy::load<Eigen::MatrixXd>(HDF5_file,
                                                                            root + obj + "/Function/ImagPart");
          auto diffraction_kochin = diffraction_kochin_real_part + MU_JJ * diffraction_kochin_imag_part;
          kochin->SetDiffractionKochin(iwaveDir, diffraction_kochin);

          // Kochin function derivative.
          if (m_hdb->GetSolver() == "Helios") {
            auto diffraction_kochin_derivative_real_part = H5Easy::load<Eigen::MatrixXd>(HDF5_file, root + obj +
                                                                                                    "/Derivative/RealPart");
            auto diffraction_kochin_derivative_imag_part = H5Easy::load<Eigen::MatrixXd>(HDF5_file, root + obj +
                                                                                                    "/Derivative/ImagPart");
            auto diffraction_kochin_derivative =
                diffraction_kochin_derivative_real_part + MU_JJ * diffraction_kochin_derivative_imag_part;
            kochin->SetDiffractionKochinDerivative(iwaveDir, diffraction_kochin_derivative);
          }
        }
        kochin->SetWaveDirectionKochin(wave_direction_kochin);

        // Radiation Kochin functions and their derivatives.
        for (int ibody = 0; ibody < m_hdb->GetNbBodies(); ++ibody) {

          // Body.
          auto body = m_hdb->GetBody(ibody);

          // Dof.
          for (unsigned int idof = 0; idof < 6; idof++) {

            // Kochin function.
            auto radiation_kochin_real_part = H5Easy::load<Eigen::MatrixXd>(
                HDF5_file, "Kochin/Radiation/Body_" + std::to_string(ibody) + "/DOF_" + std::to_string(idof)
                           + "/Function/RealPart");
            auto radiation_kochin_imag_part = H5Easy::load<Eigen::MatrixXd>(
                HDF5_file, "Kochin/Radiation/Body_" + std::to_string(ibody) + "/DOF_" + std::to_string(idof)
                           + "/Function/ImagPart");
            auto radiation_kochin = radiation_kochin_real_part + MU_JJ * radiation_kochin_imag_part;
            kochin->SetRadiationKochin(body, radiation_kochin);

            // Kochin function derivative.
            if (m_hdb->GetSolver() == "Helios") {
              auto radiation_kochin_derivative_real_part = H5Easy::load<Eigen::MatrixXd>(
                  HDF5_file, "Kochin/Radiation/Body_" + std::to_string(ibody) + "/DOF_" + std::to_string(idof)
                             + "/Derivative/RealPart");
              auto radiation_kochin_derivative_imag_part = H5Easy::load<Eigen::MatrixXd>(
                  HDF5_file, "Kochin/Radiation/Body_" + std::to_string(ibody) + "/DOF_" + std::to_string(idof)
                             + "/Derivative/ImagPart");
              auto radiation_kochin_derivative =
                  radiation_kochin_derivative_real_part + MU_JJ * radiation_kochin_derivative_imag_part;
              kochin->SetRadiationKochinDerivative(body, radiation_kochin_derivative);
            }
          }

        }
        // Add to the hdb.
        m_hdb->SetKochin(kochin);
      }
    }
  }


} // end namespace hdb5_io
