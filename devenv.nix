{ pkgs, lib, config, inputs, ... }:

let
  home = builtins.getEnv "HOME";
  source_path = "${home}/algo-trading/pysystemtrade";
  data_path = "${home}/algo-trading/pysystemtrade-data";
  config_path = "${home}/algo-trading/pysystemtrade_config";
  home_common_path = "${home}/dotfiles/nix/devenv-shell-common.nix";
  fallback_common_path = "/home/hdenman/dotfiles/nix/devenv-shell-common.nix";
  common_path =
    if builtins.pathExists home_common_path
    then home_common_path
    else fallback_common_path;
in
{
  imports = [ common_path ];

  # https://devenv.sh/basics/
  env.GREET = "pysystemtrade";

  # Runtime paths are exported in enterShell with shell-default semantics so
  # production profiles can override them before invoking `devenv shell`.



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
    export PYSYS_CODE=''${PYSYS_CODE:-${source_path}}
    export PYSYSTEMTRADE_HOME=''${PYSYSTEMTRADE_HOME:-$PYSYS_CODE}
    export PYSYS_PRIVATE_CONFIG_DIR=''${PYSYS_PRIVATE_CONFIG_DIR:-${config_path}}
    export PYTHONPATH=''${PYTHONPATH:-${source_path}}
    export SCRIPT_PATH=''${SCRIPT_PATH:-${source_path}/sysproduction/linux/scripts}
    export PARQUET_DATA=''${PARQUET_DATA:-${data_path}/parquet/}
    export MONGO_DATA=''${MONGO_DATA:-${data_path}/mongodb/}
    export MONGO_BACKUP_PATH=''${MONGO_BACKUP_PATH:-${data_path}/mongo_backup}
    export ECHO_PATH=''${ECHO_PATH:-${data_path}/echos}
    export LOG_PATH=''${LOG_PATH:-${data_path}/logs}
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
