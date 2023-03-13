// copied and modified from https://github.com/rust-lang/rust-analyzer/blob/27239fbb58a115915ffc1ce65ededc951eb00fd2/editors/code/src/commands.ts
import { LanguageClient, Location, Position } from 'vscode-languageclient/node';
import { Uri, commands } from 'vscode';

export async function showReferences(
    client: LanguageClient | undefined,
    uri: string,
    position: Position,
    locations: Location[]
) {
    if (client) {
        await commands.executeCommand(
            "editor.action.showReferences",
            Uri.parse(uri),
            client.protocol2CodeConverter.asPosition(position),
            locations.map(client.protocol2CodeConverter.asLocation)
        );
    }
}
