import { type ExtensionContext, commands, window, workspace } from "vscode";
import { LanguageClient, type LanguageClientOptions, type ServerOptions } from "vscode-languageclient/node";
import { showReferences } from "./commands";

let client: LanguageClient | undefined;

async function startLanguageClient(context: ExtensionContext) {
	try {
		const executablePath = (() => {
			const executablePath = workspace.getConfiguration("pylyzer").get<string>("executablePath", "");
			return executablePath === "" ? "pylyzer" : executablePath;
		})();
		const enableDiagnostics = workspace.getConfiguration("pylyzer").get<boolean>("diagnostics", true);
		const enableInlayHints = workspace.getConfiguration("pylyzer").get<boolean>("inlayHints", false);
		const enableSemanticTokens = workspace.getConfiguration("pylyzer").get<boolean>("semanticTokens", false);
		const enableHover = workspace.getConfiguration("pylyzer").get<boolean>("hover", true);
		const enableCompletion = workspace.getConfiguration("pylyzer").get<boolean>("completion", true);
		const smartCompletion = workspace.getConfiguration("pylyzer").get<boolean>("smartCompletion", true);
		const enableSignatureHelp = workspace.getConfiguration("pylyzer").get<boolean>("signatureHelp", true);
		const enableDocumentLink = workspace.getConfiguration("pylyzer").get<boolean>("documentLink", true);
		const enableCodeAction = workspace.getConfiguration("pylyzer").get<boolean>("codeAction", true);
		const enableCodeLens = workspace.getConfiguration("pylyzer").get<boolean>("codeLens", true);
		/* optional features */
		const checkOnType = workspace.getConfiguration("pylyzer").get<boolean>("checkOnType", false);
		const args = ["--server"];
		args.push("--");
		if (!enableDiagnostics) {
			args.push("--disable");
			args.push("diagnostic");
		}
		if (!enableInlayHints) {
			args.push("--disable");
			args.push("inlayHints");
		}
		if (!enableSemanticTokens) {
			args.push("--disable");
			args.push("semanticTokens");
		}
		if (!enableHover) {
			args.push("--disable");
			args.push("hover");
		}
		if (!enableCompletion) {
			args.push("--disable");
			args.push("completion");
		}
		if (!smartCompletion) {
			args.push("--disable");
			args.push("smartCompletion");
		}
		if (!enableSignatureHelp) {
			args.push("--disable");
			args.push("signatureHelp");
		}
		if (!enableDocumentLink) {
			args.push("--disable");
			args.push("documentLink");
		}
		if (!enableCodeAction) {
			args.push("--disable");
			args.push("codeAction");
		}
		if (!enableCodeLens) {
			args.push("--disable");
			args.push("codeLens");
		}
		if (checkOnType) {
			args.push("--enable");
			args.push("checkOnType");
		}
		const serverOptions: ServerOptions = {
			command: executablePath,
			args,
		};
		const clientOptions: LanguageClientOptions = {
			documentSelector: [
				{
					scheme: "file",
					language: "python",
				},
			],
		};
		client = new LanguageClient("pylyzer", serverOptions, clientOptions);
		await client.start();
	} catch (e) {
		window.showErrorMessage(
			"Failed to start the pylyzer language server. Please make sure you have pylyzer installed.",
		);
		window.showErrorMessage(`Error: ${e}`);
	}
}

async function restartLanguageClient() {
	try {
		if (client === undefined) {
			throw new Error();
		}
		await client.restart();
	} catch (e) {
		window.showErrorMessage("Failed to restart the pylyzer language server.");
		window.showErrorMessage(`Error: ${e}`);
	}
}

export async function activate(context: ExtensionContext) {
	context.subscriptions.push(commands.registerCommand("pylyzer.restartLanguageServer", () => restartLanguageClient()));
	context.subscriptions.push(
		commands.registerCommand("pylyzer.showReferences", async (uri, position, locations) => {
			await showReferences(client, uri, position, locations);
		}),
	);
	await startLanguageClient(context);
}

export function deactivate() {
	if (client) {
		return client.stop();
	}
}
