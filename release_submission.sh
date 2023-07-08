#!/bin/bash

rm -rf ../alpha-beta-CROWN_vnncomp23
# for testing the branch
# git clone ./ ../alpha-beta-CROWN_vnncomp23

# for actual submission
git clone git@github.com:Verified-Intelligence/Verifier_Development.git ../alpha-beta-CROWN_vnncomp23

cd ../alpha-beta-CROWN_vnncomp23
rm -rf .git
git init
git add .
git commit -m "Initial commit"
git remote add origin git@github.com:Verified-Intelligence/alpha-beta-CROWN_vnncomp23.git
git push -u --force origin master