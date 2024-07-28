# replit.nix
{ pkgs }: {
  deps = [
    pkgs.python3
    pkgs.python3Packages.pip
    pkgs.python3Packages.pytorch
    pkgs.python3Packages.torchvision
    pkgs.python3Packages.opencv4
    pkgs.python3Packages.numpy
    pkgs.python3Packages.pillow
    pkgs.python3Packages.scikit-learn
    pkgs.python3Packages.matplotlib
    pkgs.python3Packages.langdetect
    pkgs.python3Packages.imgaug
    pkgs.python3Packages.easyocr
    pkgs.python3Packages.logging
  ];
}
