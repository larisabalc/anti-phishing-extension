document.addEventListener("DOMContentLoaded", function () {
    var analyzeUrlBtn = document.getElementById("analyze-url-btn");
    var analyzeEmailBtn = document.getElementById("analyze-email-btn");
    var mainContent = document.getElementById("main-content");
    var analyzeUrlPage = document.getElementById("analyze-url-page");
    var analyzeEmailPage = document.getElementById("analyze-email-page");

    if (analyzeUrlBtn) {
        analyzeUrlBtn.addEventListener("click", function () {
            mainContent.classList.add("d-none");
            analyzeUrlPage.classList.remove("d-none");
        });
    }

    if (analyzeEmailBtn) {
        analyzeEmailBtn.addEventListener("click", function () {
            mainContent.classList.add("d-none");
            analyzeEmailPage.classList.remove("d-none");
        });
    }
});

function getQueryParam(name) {
    name = name.replace(/[\[\]]/g, "\\$&");
    var regex = new RegExp("[?&]" + name + "(=([^&#]*)|&|#|$)"),
        results = regex.exec(window.location.href);
    if (!results) return null;
    if (!results[2]) return '';
    return decodeURIComponent(results[2].replace(/\+/g, " "));
}

document.addEventListener('DOMContentLoaded', function () {
    var referrerUrl = getQueryParam('referrer');

    var referrerElement = document.getElementById('proceedLink');
    if (referrerUrl && referrerElement) {
        referrerElement.setAttribute('href', referrerUrl);
    }
});