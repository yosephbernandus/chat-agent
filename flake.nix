{
  description = "Chat Agent - RAG Chat with Ollama + ChromaDB";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        python = pkgs.python312;
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            python
            pkgs.uv
            pkgs.bun
            pkgs.process-compose
          ];

          env = {
            UV_PYTHON = "${python}/bin/python";
            LLMOBS_ENABLED = "true";
            LLMOBS_ENDPOINT = "http://localhost:8000";
            LLMOBS_API_KEY = "local-dev-key";
            LLMOBS_APP = "chat-agent";
          };

          shellHook = ''
            if [ ! -d backend/.venv ]; then
              echo "Creating backend venv..."
              uv venv backend/.venv
            fi

            if [ ! -d frontend/node_modules ]; then
              echo "Installing frontend dependencies..."
              (cd frontend && bun install)
            fi

            echo ""
            echo "Chat Agent dev shell ready"
            echo "  Run: process-compose"
            echo "  With LLMObs: LLMOBS_ENABLED=true LLMOBS_API_KEY=<key> process-compose"
            echo ""
          '';
        };
      });
}
