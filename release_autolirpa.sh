#!/bin/bash

# DEBUG=echo
private_branch=master
public_branch=master

# Cleanup.
$DEBUG rm -rf release/
$DEBUG mkdir release/

# Clone both repositories.
internal=release/internal
public=release/public
new=release/new
current=$(pwd)
$DEBUG git clone git@github.com:Verified-Intelligence/Verifier_Development.git $internal
$DEBUG git clone git@github.com:Verified-Intelligence/auto_LiRPA.git $public
# Switch branch
pushd $public
# The automatic script only releases to an internal prerelease repository
$DEBUG git remote set-url origin git@github.com:shizhouxing/auto_LiRPA_prerelease.git
$DEBUG git checkout ${public_branch}
popd

# Make a directory.
mkdir $new

# Copy other files from the internal repository.
cp -va "$internal/." $new
# Override with .git from the public repository.
rm -rf $new/.git
cp -r $public/.git $new

# Prepare for release
pushd $new
python $current/release_preprocessor.py
rm -rf complete_verifier
rm -rf vnncomp_scripts
echo "Changed files:"
# git status
# read -p "Press enter to continue, Ctrl+c to quit release."
# read -p "Now please review the diff carefully!! Press enter to view it."
# git diff
# read -p "Press enter to write commit message, Ctrl+c to quit release."
git add .
git commit -m 'September 2022 prerelease'
# read -p "Almost done. Press enter to push changes, Ctrl+c to quit release."
git push -f
popd
