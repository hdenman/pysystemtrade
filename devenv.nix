{ pkgs, lib, config, inputs, ... }:

let
  home = builtins.getEnv "HOME";
  source_path = "${home}/algo-trading/pysystemtrade";
  data_path = "${home}/algo-trading/pysystemtrade-data";
  config_path = "${home}/algo-trading/pysystemtrade_config";
in
{
  imports = [
    (builtins.getEnv "HOME" + "/dotfiles/nix/devenv-shell-common.nix")
  ];

  # https://devenv.sh/basics/
  env.GREET = "pysystemtrade";

  env.PYSYS_CODE="${source_path}";
  env.PYSYS_PRIVATE_CONFIG_DIR=config_path;
  env.PYTHONPATH = lib.mkForce "${source_path}";
  env.SCRIPT_PATH="${source_path}/sysproduction/linux/scripts";

  env.PARQUET_DATA="${data_path}/parquet/";
  env.MONGO_DATA="${data_path}/mongodb/";
  env.MONGO_BACKUP_PATH="${data_path}/mongo_backup";

  env.ECHO_PATH="${data_path}/echos";
  env.LOG_PATH="${data_path}/logs";



  # https://devenv.sh/packages/
  packages = [ pkgs.git ];

  # https://devenv.sh/languages/
  # languages.rust.enable = true;

  # https://devenv.sh/processes/
  # processes.dev.exec = "${lib.getExe pkgs.watchexec} -n -- ls -la";

  # https://devenv.sh/services/
  services.mongodb = {
    enable = true;
    package =
      if pkgs.stdenv.hostPlatform.isLinux && pkgs.stdenv.hostPlatform.isx86_64
      then
        let
          mongodWrapper = pkgs.writeShellScriptBin "mongod" ''
            #!/usr/bin/env bash
            SQLITE_DIR=""
            while [[ $# -gt 0 ]]; do
              case "$1" in
                --dbpath)
                  SQLITE_DIR="$2"
                  shift 2
                  ;;
                *)
                  shift
                  ;;
              esac
            done

            if [ -n "$SQLITE_DIR" ]; then
              mkdir -p "$SQLITE_DIR"
              exec ${pkgs.ferretdb}/bin/ferretdb --handler=sqlite --sqlite-url="file:''${SQLITE_DIR}/" --telemetry=disabled
            else
              exec ${pkgs.ferretdb}/bin/ferretdb --handler=sqlite --telemetry=disabled
            fi
          '';
        in
        pkgs.symlinkJoin {
          name = "ferretdb-mongod-wrapper";
          paths = [
            mongodWrapper
            pkgs.mongodb-tools
            pkgs.mongosh
          ];
        }
      else pkgs.mongodb-ce;
  };
  # https://devenv.sh/scripts/
  scripts.hello.exec = ''
    echo hello from $GREET
  '';

  # https://devenv.sh/basics/
  enterShell = ''
    hello
    git --version
    export OPENROUTER_API_KEY=$(cat ~/.api-keys/.openrouter-api-key-pysystemtrade)
    if [ "$(hostname)" = "marvin" ]; then
      export PYSYS_UNIVERSE=''${PYSYS_UNIVERSE:-futures}
    else
      export PYSYS_UNIVERSE=''${PYSYS_UNIVERSE:-synthetic}
    fi
    echo "Universe: $PYSYS_UNIVERSE"
  '';

  # https://devenv.sh/tasks/
  # tasks = {
  #   "myproj:setup".exec = "mytool build";
  #   "devenv:enterShell".after = [ "myproj:setup" ];
  # };

  # https://devenv.sh/tests/
  enterTest = ''
    echo "Running tests"
    git --version | grep --color=auto "${pkgs.git.version}"
  '';

  # https://devenv.sh/git-hooks/
  # git-hooks.hooks.shellcheck.enable = true;

  # See full reference at https://devenv.sh/reference/options/

  languages.python = {
    enable = true;
    version = "3.12";
    libraries = [ pkgs.zlib ];
    venv.enable = true;
    uv = {
      enable = true;
      sync.enable = true;
      sync.allPackages = true;
      sync.allGroups = true;
      sync.allExtras = true;
    };
  };
}
