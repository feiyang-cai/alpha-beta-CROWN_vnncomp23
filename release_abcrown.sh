#!/bin/bash

RELEASE_PATH="../abcrown_prerelease"  # For testing release.
# RELEASE_PATH="../alpha-beta-CROWN"  # For final release.

echo "Release criterion:"
echo "1. All private sections deleted (will be done automatically in this script."
echo "2. All GPU tests pass and all VNN-COMP 21/22 benchmarks produce expected results."
echo "3. All command examples on README and tutorial tested working."
echo "4. Remove all code files and config files that are not part of the competition, tutorial or a paper. Remove unused and unmaintained configs! Otherwise they will confuse people since many are not updated for a long time."
echo "5. Run git status and git diff in the prerelease repo and make sure changes are updated correctly. Make sure no extra unneeded files committed."
echo
echo "Current release path: ${RELEASE_PATH}"
echo
read -p "Make sure you read above carefully. Press enter to continue"

[ ! -d "${RELEASE_PATH}" ] && echo "Clone the prerelease repo before you run this script." && exit
cd "${RELEASE_PATH}"

rm -rf release
mkdir release
cd release
# git clone git@github.com:KaidiXu/Verifier_Development.git
git clone ../../Verifier_Development/.git
cd Verifier_Development
# update branch here
# git checkout fix_bound_optimizable_activation
rm -r *_release_backup
python3 release_preprocessor.py
cd ../..

git reset --hard
# Remove all files in release folder and we will copy the new version here.
rm *
git rm -r src
git rm -r complete_verifier/*
rm -r complete_verifier/*
git rm -r vnncomp_scripts/*
rm -r vnncomp_scripts/*
git rm -r auto_LiRPA/*
rm -r auto_LiRPA/*
git rm -r docs/*
rm -r docs
git rm -r examples/*
rm -r examples
git rm environment.yml LICENSE README.md setup.py

# Copy new version files.
mkdir complete_verifier
cp -r -v release/Verifier_Development/complete_verifier/* complete_verifier/
rm -r complete_verifier/auto-attack
mkdir vnncomp_scripts
cp -r -v release/Verifier_Development/vnncomp_scripts/* vnncomp_scripts
mkdir auto_LiRPA
cp -r -v release/Verifier_Development/auto_LiRPA/* auto_LiRPA
cp -v release/Verifier_Development/complete_verifier/LICENSE .
cp -v release/Verifier_Development/.gitignore .
mv release/Verifier_Development/README_abcrown.md README.md

rm -rf release/
git add -f .
git rm -r complete_verifier/exp_configs/full_dataset/* -f
git rm -r complete_verifier/exp_configs/bab_attack/attack_ubs -f
git rm -r complete_verifier/exp_configs/bab_attack/_unused -f
git rm -r complete_verifier/exp_configs/vnncomp22/before_submission -f
git rm -r complete_verifier/benchmark_data -f
git rm -r complete_verifier/cuts/CPLEX_cuts/README.md -f

