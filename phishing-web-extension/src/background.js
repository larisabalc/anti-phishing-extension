import { contextMenus, tabs, runtime} from "webextension-polyfill";

let flaggedUrls = [];

const key = new Uint8Array([
    0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
    0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10
]);
const iv = new Uint8Array([
    0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
    0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff
]);

contextMenus.create({
    id: "analyzeUrl",
    title: "Pre Check URL",
    contexts: ["link"]
});

contextMenus.onClicked.addListener(function (info, tab) {
    if (info.menuItemId === "analyzeUrl") {
        var url = info.linkUrl;
        tabs.create({ url: runtime.getURL("popup.html?referrer=" + encodeURIComponent(url)) });
    }
});

async function encryptMessage(text, key, iv) {
    const textBuffer = new TextEncoder().encode(text);

    const importedKey = await crypto.subtle.importKey(
        'raw',
        key,
        { name: 'AES-CBC' },
        false,
        ['encrypt']
    );

    const encryptedBuffer = await crypto.subtle.encrypt(
        { name: 'AES-CBC', iv },
        importedKey,
        textBuffer
    );

    const base64Encoded = btoa(String.fromCharCode.apply(null, new Uint8Array(encryptedBuffer)));

    return base64Encoded;
}

async function decryptMessage(encryptedMessage, key, iv) {
    const encryptedBuffer = new Uint8Array(atob(encryptedMessage).split('').map(c => c.charCodeAt(0)));

    const importedKey = await crypto.subtle.importKey(
        'raw',
        key,
        { name: 'AES-CBC' },
        false,
        ['decrypt']
    );

    const decryptedBuffer = await crypto.subtle.decrypt(
        { name: 'AES-CBC', iv },
        importedKey,
        encryptedBuffer
    );

    const decryptedMessage = new TextDecoder().decode(decryptedBuffer);

    return decryptedMessage;
}

async function handleMessageFromServer(event) {
    console.log('Received message from server:', event.data);
    try {
        const decryptedMessage = await decryptMessage(event.data, key, iv);
        const responseObject = JSON.parse(decryptedMessage);

        if (responseObject.result === 'malicious' && responseObject.type === 'CurrentTab' && !flaggedUrls.includes(responseObject.url)) {
            console.log('Phishy activity!');
            flaggedUrls.push(responseObject.url);
            tabs.create({ url: runtime.getURL("danger.html?referrer=" + encodeURIComponent(responseObject.url)) });
        }
        runtime.sendMessage(responseObject);
    } catch (error) {
        console.error('Error processing WebSocket message:', error);
    }
}

async function sendMessage(socket, jsonObject) {
    const json = JSON.stringify(jsonObject);
    const encrypted_json = await encryptMessage(json, key, iv);
    socket.send(encrypted_json);
}

function handleMessageFromPopup(message, sender, sendResponse) {
    const socket = new WebSocket("ws://localhost:9999");

    socket.onopen = async function (event) {
        console.log('WebSocket connection established.');

        if (message.type === 'CurrentTab' || message.type === 'PreCheck') {
            const jsonObject = {
                url: message.urls,
                type: message.type,
            };
            await sendMessage(socket, jsonObject);
        } else if (message.type === 'FetchUrls') {
            if(message.urls) {
                const jsonObject = {
                    urls: message.urls,
                    type: 'FetchUrls',
                }
                await sendMessage(socket, jsonObject);
            }
        } else if (message.type === 'CheckEmail') {
            const jsonObject = {
                email: message.email,
                type: 'CheckEmail',
            };
            await sendMessage(socket, jsonObject);
        } else if (message.type === 'FalseNegative') {
            const jsonObject = {
                url: message.url,
                type: message.type,
                result: message.result,
            };
            await sendMessage(socket, jsonObject);
        }
    };

    socket.onmessage = function (event) {
        handleMessageFromServer(event);
    };

    socket.onerror = function (error) {
        console.error('WebSocket error:', error);
    };
}

runtime.onMessage.addListener(handleMessageFromPopup);
