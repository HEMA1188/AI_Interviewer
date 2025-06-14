// static/app.js

document.addEventListener('DOMContentLoaded', () => {
    console.log("App.js loaded. DOM Content Loaded.");

    // --- DOM Elements ---
    const startInterviewBtn = document.getElementById('startInterviewBtn');
    const interviewSetupSection = document.getElementById('interviewSetup');
    const interviewSection = document.getElementById('interviewSection');
    const interviewControls = document.getElementById('interviewControls'); // Contains question/answer
    const statusMessage = document.getElementById('statusMessage');
    const currentQuestionDisplay = document.getElementById('currentQuestionDisplay');
    const candidateAnswerInput = document.getElementById('candidateAnswer');
    const submitAnswerBtn = document.getElementById('submitAnswerBtn');
    const aiAvatarVideo = document.getElementById('aiAvatarVideo');
    const interviewTranscript = document.getElementById('interviewTranscript');
    const endInterviewBtn = document.getElementById('endInterviewBtn');
    const finalReportSection = document.getElementById('finalReportSection');
    const finalReportDisplay = document.getElementById('finalReportDisplay');
    const startNewInterviewBtn = document.getElementById('startNewInterviewBtn');

    // Setup Inputs
    const candidateNameInput = document.getElementById('candidateName');
    const resumeTextInput = document.getElementById('resumeText');
    const experienceLevelSelect = document.getElementById('experienceLevel'); // Fresher/Experience dropdown
    const interviewLevelSelect = document.getElementById('interviewLevel');   // Basic/Intermediate/Advanced dropdown

    // --- State Variables ---
    let currentTalkId = null;
    let pollingIntervalId = null;
    let currentQuestionText = ""; // Store the current question asked by AI for follow-up logic
    let questionCount = 0; // To track number of questions asked

    // --- API Base URL (adjust if your backend is on a different port/host) ---
    const API_BASE_URL = window.location.origin; // Assumes backend is on the same origin

    // --- Utility Functions ---

    /** Displays messages to the user. */
    function displayStatus(message, isError = false) {
        statusMessage.textContent = message;
        statusMessage.style.color = isError ? 'red' : 'green';
        if (message) {
             statusMessage.classList.remove('hidden');
        } else {
             statusMessage.classList.add('hidden');
        }
    }

    /** Appends a message to the transcript. */
    function appendToTranscript(speaker, text, isVideo = false) {
        const p = document.createElement('p');
        p.innerHTML = `<strong>${speaker}:</strong> ${text}`;
        if (isVideo) {
            p.innerHTML += ' <small>(Playing video...)</small>';
        }
        interviewTranscript.appendChild(p);
        interviewTranscript.scrollTop = interviewTranscript.scrollHeight; // Scroll to bottom
    }

    /** Disables/enables UI controls. */
    function setControlsEnabled(enabled) {
        submitAnswerBtn.disabled = !enabled;
        candidateAnswerInput.disabled = !enabled;
        endInterviewBtn.disabled = !enabled;
    }

    /** Plays the D-ID video and updates UI. */
    function playDIdVideo(videoUrl) {
        if (videoUrl) {
            aiAvatarVideo.src = videoUrl;
            aiAvatarVideo.style.display = 'block'; // Show video element
            aiAvatarVideo.play().catch(error => {
                console.error("Error playing video:", error);
                displayStatus("Error playing avatar video. Check console.", true);
            });
            // Hide the video when it ends or fails
            aiAvatarVideo.onended = () => {
                aiAvatarVideo.pause();
                aiAvatarVideo.style.display = 'none'; // Hide video element
                aiAvatarVideo.src = ''; // Clear source
            };
            aiAvatarVideo.onerror = (e) => {
                console.error("Video element error:", e);
                aiAvatarVideo.style.display = 'none'; // Hide video element
                aiAvatarVideo.src = ''; // Clear source
            };
        } else {
            aiAvatarVideo.style.display = 'none'; // Ensure video is hidden if no URL
            aiAvatarVideo.src = '';
        }
    }

    /** Polls the backend for D-ID video status. */
    async function pollForDIdVideo(talkId) {
        if (!talkId) {
            console.warn("No D-ID talk ID to poll.");
            return null;
        }

        displayStatus("Generating avatar video...");
        setControlsEnabled(false); // Disable controls while video is generating

        let status = "";
        let videoUrl = null;
        let attempts = 0;
        const maxAttempts = 40; // Max 80 seconds at 2-sec interval
        const pollInterval = 2000; // 2 seconds

        return new Promise(async (resolve) => {
            pollingIntervalId = setInterval(async () => {
                attempts++;
                if (attempts > maxAttempts) {
                    clearInterval(pollingIntervalId);
                    displayStatus("Video generation timed out. Showing text only.", true);
                    setControlsEnabled(true);
                    resolve(null);
                    return;
                }

                console.log(`Polling D-ID status for ${talkId}, attempt ${attempts}...`);
                try {
                    const response = await fetch(`${API_BASE_URL}/api/d_id_status?talk_id=${talkId}`);
                    const data = await response.json();

                    status = data.status;
                    videoUrl = data.result_url;

                    if (status === "done") {
                        clearInterval(pollingIntervalId);
                        displayStatus("Video ready!");
                        setControlsEnabled(true);
                        resolve(videoUrl);
                    } else if (status === "error") {
                        clearInterval(pollingIntervalId);
                        displayStatus(`Video generation failed: ${data.details || data.error}`, true);
                        setControlsEnabled(true);
                        resolve(null);
                    } else {
                        displayStatus(`Video status: ${status}...`);
                    }
                } catch (error) {
                    clearInterval(pollingIntervalId);
                    console.error("Error polling D-ID status:", error);
                    displayStatus("Error checking video status. Please check console.", true);
                    setControlsEnabled(true);
                    resolve(null);
                }
            }, pollInterval);
        });
    }

    /** Handles AI responses (text and video). */
    async function handleAIResponse(data, speaker = "AI Interviewer") {
        const aiText = data.text;
        const dIdTalkId = data.d_id_talk_id;

        appendToTranscript(speaker, aiText, dIdTalkId !== null);

        if (dIdTalkId) {
            currentTalkId = dIdTalkId; // Store for potential future reference
            const videoUrl = await pollForDIdVideo(dIdTalkId);
            if (videoUrl) {
                playDIdVideo(videoUrl);
                // Optionally, hide text when video starts playing and show when ends
                currentQuestionDisplay.textContent = aiText; // Always update question display
            } else {
                console.warn("Could not get D-ID video, displaying text fallback.");
                currentQuestionDisplay.textContent = aiText;
            }
        } else {
            console.warn("No D-ID talk ID received for this response. Displaying text fallback.");
            currentQuestionDisplay.textContent = aiText;
            setControlsEnabled(true); // Ensure controls are enabled if no video to wait for
        }
    }


    // --- Event Listeners ---

    startInterviewBtn.addEventListener('click', async () => {
        const candidateName = candidateNameInput.value.trim();
        const resumeText = resumeTextInput.value.trim();
        const experienceLevel = experienceLevelSelect.value;
        const interviewLevel = interviewLevelSelect.value;

        if (!candidateName || !resumeText || experienceLevel === "" || interviewLevel === "") {
            displayStatus("Please fill in all setup fields.", true);
            return;
        }

        displayStatus("Starting interview session...");
        startInterviewBtn.disabled = true; // Disable button to prevent multiple clicks

        try {
            const response = await fetch(`${API_BASE_URL}/api/start_interview`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    candidate_name: candidateName,
                    resume_text: resumeText,
                    candidate_actual_experience: experienceLevel,
                    chosen_interview_level: interviewLevel
                })
            });
            const data = await response.json();

            if (response.ok) {
                console.log("Interview session started:", data);
                interviewSetupSection.classList.add('hidden');
                interviewSection.classList.remove('hidden');
                interviewControls.classList.remove('hidden');
                finalReportSection.classList.add('hidden'); // Ensure final report is hidden

                // Now get the greeting
                displayStatus("Generating greeting...");
                const greetingResponse = await fetch(`${API_BASE_URL}/api/get_greeting`);
                const greetingData = await greetingResponse.json();

                if (greetingResponse.ok) {
                    handleAIResponse(greetingData, "AI Interviewer");
                    questionCount = 0; // Reset question count
                    currentQuestionText = greetingData.text; // Store greeting as current 'question' for history
                    setControlsEnabled(true);
                } else {
                    displayStatus(`Error getting greeting: ${greetingData.error || 'Unknown error'}`, true);
                    setControlsEnabled(false);
                }

            } else {
                displayStatus(`Error starting interview: ${data.error || 'Unknown error'}`, true);
                startInterviewBtn.disabled = false;
            }
        } catch (error) {
            console.error("Network or server error:", error);
            displayStatus("Network or server error. Please try again.", true);
            startInterviewBtn.disabled = false;
        }
    });

    submitAnswerBtn.addEventListener('click', async () => {
        const candidateAnswer = candidateAnswerInput.value.trim();
        if (!candidateAnswer) {
            displayStatus("Please enter your answer.", true);
            return;
        }

        appendToTranscript("You", candidateAnswer);
        displayStatus("Processing your answer and generating next question...");
        setControlsEnabled(false); // Disable controls while processing

        try {
            const response = await fetch(`${API_BASE_URL}/api/submit_answer`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    answer_text: candidateAnswer,
                    //You might need to send the previous question here if your backend uses it for follow-up logic
                    //previous_question: currentQuestionText 
                })
            });
            const data = await response.json();

            if (response.ok) {
                candidateAnswerInput.value = ''; // Clear input
                questionCount++;

                // The backend can decide if it's a new question or just an acknowledgement
                // The `get_next_question` endpoint will be specifically for new questions.
                // `submit_answer` can simply process the answer and update history.
                // Then, you'd likely call `get_next_question` again.

                // For simplicity, let's assume submit_answer directly triggers the next AI response
                // which could be a follow-up, a new question, or a transition.
                if (data.next_response_text || data.d_id_talk_id) { // Check if there's an actual response from AI
                    handleAIResponse(data, "AI Interviewer");
                    currentQuestionText = data.text; // Update current question text
                } else {
                    // If no direct response, maybe just a backend processing message
                    displayStatus("Answer submitted. Awaiting next question...", false);
                    // Manually trigger next question generation if submit_answer doesn't return it
                    await getNextQuestion();
                }

            } else {
                displayStatus(`Error submitting answer: ${data.error || 'Unknown error'}`, true);
                setControlsEnabled(true);
            }
        } catch (error) {
            console.error("Network or server error:", error);
            displayStatus("Network or server error. Please try again.", true);
            setControlsEnabled(true);
        }
    });

    endInterviewBtn.addEventListener('click', async () => {
        displayStatus("Ending interview and generating report...");
        setControlsEnabled(false);

        // Clear any active polling interval
        if (pollingIntervalId) {
            clearInterval(pollingIntervalId);
            pollingIntervalId = null;
        }
        
        try {
            // First, get the closing statement
            const closingResponse = await fetch(`${API_BASE_URL}/api/get_closing`);
            const closingData = await closingResponse.json();

            if (closingResponse.ok) {
                await handleAIResponse(closingData, "AI Interviewer"); // Wait for video to process
            } else {
                displayStatus(`Error getting closing: ${closingData.error || 'Unknown error'}`, true);
            }

            // Then, get the final report
            const reportResponse = await fetch(`${API_BASE_URL}/api/get_final_report`);
            const reportData = await reportResponse.json();

            if (reportResponse.ok) {
                finalReportDisplay.textContent = reportData.report_text;
                interviewSection.classList.add('hidden');
                finalReportSection.classList.remove('hidden');
                displayStatus("Interview ended. Report generated.");
            } else {
                displayStatus(`Error generating final report: ${reportData.error || 'Unknown error'}`, true);
            }

        } catch (error) {
            console.error("Network or server error during end interview:", error);
            displayStatus("Network or server error during end interview. Please try again.", true);
            setControlsEnabled(true);
        }
    });

    startNewInterviewBtn.addEventListener('click', () => {
        // Reset all UI and state
        interviewSetupSection.classList.remove('hidden');
        interviewSection.classList.add('hidden');
        finalReportSection.classList.add('hidden');
        interviewControls.classList.add('hidden');

        candidateNameInput.value = '';
        resumeTextInput.value = '';
        experienceLevelSelect.value = '';
        interviewLevelSelect.value = '';
        candidateAnswerInput.value = '';
        currentQuestionDisplay.textContent = '';
        interviewTranscript.innerHTML = ''; // Clear transcript
        finalReportDisplay.textContent = '';
        statusMessage.textContent = ''; // Clear status

        currentTalkId = null;
        if (pollingIntervalId) {
            clearInterval(pollingIntervalId);
            pollingIntervalId = null;
        }
        currentQuestionText = "";
        questionCount = 0;
        startInterviewBtn.disabled = false; // Re-enable start button
        setControlsEnabled(false); // Ensure controls are disabled initially
    });

    // --- Initial State Setup ---
    setControlsEnabled(false); // Controls are disabled until interview starts
    displayStatus(""); // Clear initial status message
    aiAvatarVideo.style.display = 'none'; // Hide video initially
    finalReportSection.classList.add('hidden'); // Ensure hidden at start
    interviewSection.classList.add('hidden'); // Ensure hidden at start
    interviewControls.classList.add('hidden'); // Ensure hidden at start

}); // End DOMContentLoaded