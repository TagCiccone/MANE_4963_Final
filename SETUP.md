1. Ensure you have BeamNG.tech installed on the host windows machine
2. Ensure this repository is located within your WSL filesystem
3. Within WSL, run `echo "export WINDOWS_USERNAME=$(printf '"%s"\n' "$(cmd.exe /c echo %USERNAME% 2>/dev/null | tr -d '\r')")" >> ~/.bashrc"`
4. Run the dev container