"use strict";
const vscode = require("vscode");
const languageclient = require("vscode-languageclient");

let client;

function activate(context) {
    try {
        const serverOptions = {
            command: "pype",
            args: ["--server"]
        };
        const clientOptions = {
            documentSelector: [
                {
                    scheme: "file",
                    language: "python",
                }
            ],
        };
        client = new languageclient.LanguageClient("pype", serverOptions, clientOptions);
        context.subscriptions.push(client.start());
    } catch (e) {
        vscode.window.showErrorMessage("failed to start pype.");
    }
}

function deactivate() {
    if (client) return client.stop();
}

module.exports = { activate, deactivate }
