{
  description = "Nix Flake for Pylyzer";

  # Flake inputs
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

    # develop
    devshell.url = "github:numtide/devshell";
    flake-utils.url = "github:numtide/flake-utils";
    flake-compat = {
      url = "github:edolstra/flake-compat";
      flake = false;
    };

    rust-overlay.url = "github:oxalica/rust-overlay"; # A helper for Rust + Nix
  };
  outputs = {
    self,
    nixpkgs,
    devshell,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
        overlays = [devshell.overlays.default];
      };
      cargoToml = with builtins; (fromTOML (readFile ./Cargo.toml));

      inherit (pkgs) lib;
    in {
      packages.pylyzer = pkgs.rustPlatform.buildRustPackage {
        inherit (cargoToml.package) name;
        inherit (cargoToml.workspace.package) version;
        src = builtins.path {
          path = ./.;
          filter = name: type:
            (name == toString ./Cargo.toml)
            || (name == toString ./Cargo.lock)
            || (lib.hasPrefix (toString ./crates) name)
            || (lib.hasPrefix (toString ./src) name);
        };
        cargoLock.lockFile = ./Cargo.lock;
      };
      packages.default = self.packages.${system}.erg;

      devShells.default = pkgs.devshell.mkShell {
        packages = with pkgs; [
          rustc
          cargo
          # Dev
          python3
          alejandra # Nix formatter
          rustfmt # Rust Formatter
          taplo-cli # TOML formatter
        ];
      };

      checks = {pylyzer = self.packages.${system}.pylyzer;};
    });
}
