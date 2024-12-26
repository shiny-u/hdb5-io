# Changelog

This is the Changelog of the FRyDom framework.
This file should be kept up to date following [these guidelines](https://keepachangelog.com/en/1.0.0/)

## [Unrelease]

### Added 

- Orcaflex reader for multibody configuration in HDB5tool

### Changed

### Fixed

- Enable HDB5 loader without times discretization and infinite added mass data
- Orcaflex added mass and damping reader in HDB5tool
- Mask initialization with orcaflex reader in HDB5tool
- Float precision with last numpy versions

## [3.2] 2024-02-22

### Added

- Add orcaflex hdb reader


## [3.1.2] 2024-02-16

### Changed

- Remove RAO limit with w_max hardcoded 

### Fixed

- Wave frequeny discretization in HDB5Tool command argument

## [3.1.1] 2023-11-30

### Added 

- Add extra damping linear termes in input arguments

### Changed

- Remove RAO limit with w_max hardcoded 

### Fixed

- Add pi/2 phase in diodore input reader
- RAO is optional in diodore input reader
- Wave frequeny discretization in HDB5Tool command argument

## [3.1] 2023-11-13

### Added 

- Get BEMBody by name 

### Fixed

- HDB5tool : warning message when trying to build rao with null hydrostatic matrix

## [3.0] 2023-10-16

### Added
- Improve error message and warning when computing RAO without mass or hydrostatics.

### Changed
- update hdb5_io and HDBtool to HDB version 4 (wave drift attached to body)
- Remove unused old Diodore HDB reader
- Remove backward compability with older HDB version

## [2.9.0] 2023-09-18

### Added
- HDB5tool : python script for HDB5 database manipulation

### Fixed
- Enable HDB5 reader without KochinStep in hdyrodynamic database
