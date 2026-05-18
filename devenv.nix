{ pkgs, lib, config, inputs, ... }:

let
  source_path = "/Users/hdenman/workspace/pysystemtrade";
  data_path = "/Users/hdenman/pysystemtrade-data";
in
{
  # https://devenv.sh/basics/
  env.GREET = "pysystemtrade";
  env.PYSYS_CODE="${source_path}";
  env.SCRIPT_PATH="${source_path}/sysproduction/linux/scripts";

  env.MONGO_DATA="${data_path}/mongodb/";
  env.MONGO_BACKUP_PATH="${data_path}/mongo_backup";
  env.ECHO_PATH="${data_path}/echos";

  # https://devenv.sh/packages/
  packages = [ pkgs.git ];

  # https://devenv.sh/languages/
  # languages.rust.enable = true;

  # https://devenv.sh/processes/
  # processes.dev.exec = "${lib.getExe pkgs.watchexec} -n -- ls -la";

  # https://devenv.sh/services/
  services.mongodb = {
    enable = true;
  };

  # https://devenv.sh/scripts/
  scripts.hello.exec = ''
    echo hello from $GREET
  '';

  # https://devenv.sh/basics/
  enterShell = ''
    hello         # Run scripts directly
    git --version # Use packages
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
