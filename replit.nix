{ pkgs }: {
  deps = [
    pkgs.python310Full
    pkgs.python310Packages.pip
    pkgs.python310Packages.numpy
    pkgs.python310Packages.opencv4
    pkgs.python310Packages.pillow
    pkgs.python310Packages.matplotlib
    pkgs.python310Packages.scikitlearn
    pkgs.python310Packages.tqdm
    pkgs.python310Packages.joblib
    pkgs.libGL
    pkgs.libGLU
    pkgs.glib
    pkgs.xorg.libX11
    pkgs.xorg.libXext
    pkgs.xorg.libXrender
    pkgs.xorg.libXi
    pkgs.xorg.libXfixes
    pkgs.xorg.libXcursor
  ];
  env = {
    PYTHON_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib
      pkgs.glib
      pkgs.libGL
      pkgs.libGLU
    ];
    PYTHONBIN = "${pkgs.python310Full}/bin/python3.10";
    LANG = "en_US.UTF-8";
  };
}
