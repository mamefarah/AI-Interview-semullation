
import React, { useReducer, useRef, useCallback, useEffect } from 'react';
import { GoogleGenAI, Modality, LiveServerMessage } from "@google/genai";
import * as pdfjsLib from 'pdfjs-dist/build/pdf.min.mjs';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { InterviewState, TranscriptEntry } from './types';
import { generateInterviewSummary, generateWelcomeAudio } from './services/geminiService';
import { decode, decodeAudioData, createBlob, calculateRMS } from './utils/audio';
import { PlayIcon, StopIcon, RobotIcon, UserIcon, UploadIcon, MicIcon, SpeakerIcon } from './components/IconComponents';
import { getApiErrorMessage } from './utils/error';

const languageOptions = [
    { value: 'English', label: 'English' },
    { value: 'Spanish', label: 'Español' },
    { value: 'French', label: 'Français' },
    { value: 'German', label: 'Deutsch' },
    { value: 'Japanese', label: '日本語' },
    { value: 'Mandarin Chinese', label: '普通话' },
    { value: 'Hindi', label: 'हिन्दी' },
    { value: 'Portuguese', label: 'Português' },
];

const SILENCE_THRESHOLD = 0.01; // Sensitivity for silence detection.
const SILENCE_DURATION_MS = 1500; // 1.5 seconds of silence to trigger end of turn.

interface AppState {
    script: string;
    interviewState: InterviewState;
    transcript: TranscriptEntry[];
    currentInterviewerText: string;
    currentUserText: string;
    summary: string | null;
    error: string | null;
    language: string;
    micGain: number;
    speakerVolume: number;
}

type AppAction =
    | { type: 'SET_STATE'; payload: Partial<AppState> }
    | { type: 'START_CONNECTING'; payload: { script: string; language: string } }
    | { type: 'CONNECTION_ESTABLISHED' }
    | { type: 'UPDATE_TRANSCRIPTIONS'; payload: { user: string; interviewer: string } }
    | { type: 'FINALIZE_TURN'; payload: { user: string; interviewer: string } }
    | { type: 'START_SUMMARIZING' }
    | { type: 'FINISH_INTERVIEW'; payload: string }
    | { type: 'SET_ERROR'; payload: string }
    | { type: 'RESET' };

const initialState: AppState = {
    script: "Conduct a 15-minute interview for a senior product manager role. Focus on strategy, execution, and leadership. Start by introducing yourself.",
    interviewState: InterviewState.IDLE,
    transcript: [],
    currentInterviewerText: '',
    currentUserText: '',
    summary: null,
    error: null,
    language: 'English',
    micGain: 1,
    speakerVolume: 1,
};

function appReducer(state: AppState, action: AppAction): AppState {
    switch (action.type) {
        case 'SET_STATE':
            return { ...state, ...action.payload };
        case 'START_CONNECTING':
            return {
                ...state,
                script: action.payload.script,
                language: action.payload.language,
                interviewState: InterviewState.CONNECTING,
                error: null,
                transcript: [],
                currentInterviewerText: '',
                currentUserText: '',
                summary: null,
            };
        case 'CONNECTION_ESTABLISHED':
            return { ...state, interviewState: InterviewState.IN_CONVERSATION };
        case 'UPDATE_TRANSCRIPTIONS':
            return {
                ...state,
                currentUserText: action.payload.user,
                currentInterviewerText: action.payload.interviewer,
            };
        case 'FINALIZE_TURN':
            const newTranscript = [...state.transcript];
            if (action.payload.user) newTranscript.push({ speaker: 'You', text: action.payload.user });
            if (action.payload.interviewer) newTranscript.push({ speaker: 'Interviewer', text: action.payload.interviewer });
            return {
                ...state,
                transcript: newTranscript,
                currentUserText: '',
                currentInterviewerText: '',
            };
        case 'START_SUMMARIZING':
            return { ...state, interviewState: InterviewState.SUMMARIZING };
        case 'FINISH_INTERVIEW':
            return { ...state, interviewState: InterviewState.FINISHED, summary: action.payload };
        case 'SET_ERROR':
            return { ...state, interviewState: InterviewState.ERROR, error: action.payload };
        case 'RESET':
            return initialState;
        default:
            return state;
    }
}

const AudioControls = ({ micGain, onMicGainChange, speakerVolume, onSpeakerVolumeChange }: {
    micGain: number;
    onMicGainChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
    speakerVolume: number;
    onSpeakerVolumeChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}) => (
    <div className="flex flex-col sm:flex-row gap-6">
        <div className="flex-1">
            <label htmlFor="mic-gain" className="flex items-center text-sm font-medium text-gray-300 mb-2">
                <MicIcon className="h-5 w-5 mr-2" /> Microphone Sensitivity
            </label>
            <input
                id="mic-gain"
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={micGain}
                onChange={onMicGainChange}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
                aria-label="Microphone Sensitivity"
            />
        </div>
        <div className="flex-1">
            <label htmlFor="speaker-volume" className="flex items-center text-sm font-medium text-gray-300 mb-2">
                <SpeakerIcon className="h-5 w-5 mr-2" /> Speaker Volume
            </label>
            <input
                id="speaker-volume"
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={speakerVolume}
                onChange={onSpeakerVolumeChange}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
                aria-label="Speaker Volume"
            />
        </div>
    </div>
);


const App: React.FC = () => {
    const [state, dispatch] = useReducer(appReducer, initialState);
    const { script, interviewState, transcript, currentInterviewerText, currentUserText, summary, error, language, micGain, speakerVolume } = state;

    const sessionPromise = useRef<Promise<any> | null>(null);
    const inputAudioContext = useRef<AudioContext | null>(null);
    const outputAudioContext = useRef<AudioContext | null>(null);
    const scriptProcessorNode = useRef<ScriptProcessorNode | null>(null);
    const mediaStreamSource = useRef<MediaStreamAudioSourceNode | null>(null);
    const inputGainNode = useRef<GainNode | null>(null);
    const outputGainNode = useRef<GainNode | null>(null);
    const userMediaStream = useRef<MediaStream | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const transcriptEndRef = useRef<HTMLDivElement>(null);

    // Fix: Use ReturnType<typeof setTimeout> for the timer reference to ensure browser compatibility.
    const silenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const hasSpokenInTurnRef = useRef(false);
    const isSilentRef = useRef(false);

    const currentInterviewerTextRef = useRef('');
    const currentUserTextRef = useRef('');
    
    const audioSources = useRef<Set<AudioBufferSourceNode>>(new Set());
    const nextStartTime = useRef(0);

    useEffect(() => {
        try {
            pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdn.jsdelivr.net/npm/pdfjs-dist@4.4.168/build/pdf.worker.min.mjs';
        } catch (e) {
            console.error("Failed to set PDF.js worker source.", e);
        }
    }, []);

    useEffect(() => {
        transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [transcript, currentInterviewerText, currentUserText]);

    const cleanupAudio = useCallback(() => {
        if (silenceTimerRef.current) {
            clearTimeout(silenceTimerRef.current);
            silenceTimerRef.current = null;
        }
        if (sessionPromise.current) {
            sessionPromise.current.then(session => session.close()).catch(console.error);
            sessionPromise.current = null;
        }
        scriptProcessorNode.current?.disconnect();
        mediaStreamSource.current?.disconnect();
        inputGainNode.current?.disconnect();
        outputGainNode.current?.disconnect();
        userMediaStream.current?.getTracks().forEach(track => track.stop());
        inputAudioContext.current?.close().catch(console.error);
        outputAudioContext.current?.close().catch(console.error);

        audioSources.current.forEach(source => source.stop());
        audioSources.current.clear();
        nextStartTime.current = 0;
        
        scriptProcessorNode.current = null;
        mediaStreamSource.current = null;
        inputGainNode.current = null;
        outputGainNode.current = null;
        userMediaStream.current = null;
        inputAudioContext.current = null;
        outputAudioContext.current = null;
    }, []);

    const handleStopConversation = useCallback(async () => {
        cleanupAudio();
        dispatch({ type: 'START_SUMMARIZING' });
        try {
            const finalTranscript: TranscriptEntry[] = [
                ...transcript,
                ...(currentUserTextRef.current.trim() ? [{ speaker: 'You', text: currentUserTextRef.current.trim() }] : []),
                ...(currentInterviewerTextRef.current.trim() ? [{ speaker: 'Interviewer', text: currentInterviewerTextRef.current.trim() }] : []),
            ];
            const result = await generateInterviewSummary(finalTranscript, language);
            dispatch({ type: 'FINISH_INTERVIEW', payload: result });
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : 'Failed to generate summary.';
            dispatch({ type: 'SET_ERROR', payload: message });
        }
    }, [cleanupAudio, transcript, language]);

    const handleStartConversation = useCallback(async () => {
        if (script.trim() === '') {
            dispatch({ type: 'SET_ERROR', payload: "Please provide an interview topic or script." });
            return;
        }
        
        dispatch({ type: 'START_CONNECTING', payload: { script, language } });
        currentInterviewerTextRef.current = '';
        currentUserTextRef.current = '';

        try {
            if (!process.env.API_KEY) throw new Error("API key not configured.");
            const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
            
            outputAudioContext.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
            outputGainNode.current = outputAudioContext.current.createGain();
            outputGainNode.current.gain.value = speakerVolume;
            outputGainNode.current.connect(outputAudioContext.current.destination);
            nextStartTime.current = 0;

            generateWelcomeAudio(language).then(async (welcomeAudioBase64) => {
                if (welcomeAudioBase64 && outputAudioContext.current && outputGainNode.current) {
                    try {
                        const audioBuffer = await decodeAudioData(decode(welcomeAudioBase64), outputAudioContext.current, 24000, 1);
                        const source = outputAudioContext.current.createBufferSource();
                        source.buffer = audioBuffer;
                        source.connect(outputGainNode.current);
                        source.start();
                    } catch (audioError) {
                        console.error("Failed to play welcome audio:", audioError);
                    }
                }
            });

            userMediaStream.current = await navigator.mediaDevices.getUserMedia({ audio: true });
            inputAudioContext.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
            inputGainNode.current = inputAudioContext.current.createGain();
            inputGainNode.current.gain.value = micGain;
            
            sessionPromise.current = ai.live.connect({
                model: 'gemini-2.5-flash-native-audio-preview-09-2025',
                callbacks: {
                    onopen: () => {
                        dispatch({ type: 'CONNECTION_ESTABLISHED' });
                        const source = inputAudioContext.current!.createMediaStreamSource(userMediaStream.current!);
                        mediaStreamSource.current = source;
                        const processor = inputAudioContext.current!.createScriptProcessor(4096, 1, 1);
                        scriptProcessorNode.current = processor;
                        
                        processor.onaudioprocess = (audioProcessingEvent) => {
                            const inputData = audioProcessingEvent.inputBuffer.getChannelData(0);
                            const rms = calculateRMS(inputData);
                            if (rms > SILENCE_THRESHOLD) {
                                // User is speaking
                                if (silenceTimerRef.current) {
                                    clearTimeout(silenceTimerRef.current);
                                    silenceTimerRef.current = null;
                                }
                                isSilentRef.current = false;
                            } else {
                                // User is silent
                                if (hasSpokenInTurnRef.current && !silenceTimerRef.current) {
                                    silenceTimerRef.current = setTimeout(() => {
                                        // Silence timeout reached, assume end of turn by stopping audio stream
                                        isSilentRef.current = true;
                                    }, SILENCE_DURATION_MS);
                                }
                            }
                            
                            if (!isSilentRef.current) {
                                const pcmBlob = createBlob(inputData);
                                sessionPromise.current?.then((session) => {
                                    session.sendRealtimeInput({ media: pcmBlob });
                                });
                            }
                        };
                        source.connect(inputGainNode.current!);
                        inputGainNode.current!.connect(processor);
                        processor.connect(inputAudioContext.current!.destination);
                    },
                    onmessage: async (message: LiveServerMessage) => {
                        const base64Audio = message.serverContent?.modelTurn?.parts[0]?.inlineData.data;
                        if (base64Audio && outputAudioContext.current && outputGainNode.current) {
                            nextStartTime.current = Math.max(nextStartTime.current, outputAudioContext.current.currentTime);
                            const audioBuffer = await decodeAudioData(decode(base64Audio), outputAudioContext.current, 24000, 1);
                            const source = outputAudioContext.current.createBufferSource();
                            source.buffer = audioBuffer;
                            source.connect(outputGainNode.current);
                            source.addEventListener('ended', () => audioSources.current.delete(source));
                            source.start(nextStartTime.current);
                            nextStartTime.current += audioBuffer.duration;
                            audioSources.current.add(source);
                        }

                        if (message.serverContent?.interrupted) {
                           audioSources.current.forEach(source => source.stop());
                           audioSources.current.clear();
                           nextStartTime.current = 0;
                        }

                        if (message.serverContent?.outputTranscription) {
                            currentInterviewerTextRef.current += message.serverContent.outputTranscription.text;
                        }
                        if (message.serverContent?.inputTranscription) {
                            currentUserTextRef.current += message.serverContent.inputTranscription.text;
                            if (message.serverContent.inputTranscription.text.trim()) {
                                hasSpokenInTurnRef.current = true;
                            }
                        }
                        dispatch({ type: 'UPDATE_TRANSCRIPTIONS', payload: { user: currentUserTextRef.current, interviewer: currentInterviewerTextRef.current }});
                        
                        if (message.serverContent?.turnComplete) {
                            dispatch({ type: 'FINALIZE_TURN', payload: { user: currentUserTextRef.current.trim(), interviewer: currentInterviewerTextRef.current.trim() }});
                            currentUserTextRef.current = '';
                            currentInterviewerTextRef.current = '';

                             // Reset silence detection state for the next turn
                            hasSpokenInTurnRef.current = false;
                            isSilentRef.current = false; // Allow audio streaming to resume
                            if (silenceTimerRef.current) {
                                clearTimeout(silenceTimerRef.current);
                                silenceTimerRef.current = null;
                            }
                        }
                    },
                    onerror: (e: ErrorEvent) => {
                        console.error('Live session error:', e);
                        dispatch({ type: 'SET_ERROR', payload: getApiErrorMessage(e, 'start') });
                        cleanupAudio();
                    },
                    onclose: () => {},
                },
                config: {
                    systemInstruction: `You are a professional interviewer. Your voice should be clear and engaging. Conduct an interview in ${language} based on these instructions: "${script}". Do not mention that you are an AI. Begin the interview directly without any introduction or welcome message.`,
                    responseModalities: [Modality.AUDIO],
                    inputAudioTranscription: {},
                    outputAudioTranscription: {},
                    speechConfig: {
                        voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } },
                    },
                },
            });

        } catch (err: unknown) {
            console.error("Failed to start conversation:", err);
            let errorMessage: string;

            if (err instanceof Error) {
                if (err.message === "API key not configured.") {
                    errorMessage = "The API key is not configured. This is a setup issue and requires developer attention.";
                } else if (err.name === 'NotAllowedError') {
                    errorMessage = "Microphone access was denied. Please enable microphone permissions for this site in your browser settings and try again.";
                } else if (err.name === 'NotFoundError') {
                    errorMessage = "No microphone was found on your device. Please connect a microphone and try again.";
                } else if (err.name === 'NotReadableError') {
                    errorMessage = "There was a hardware error with your microphone, or it might be in use by another application. Please check your microphone and try again.";
                } else {
                    errorMessage = getApiErrorMessage(err, 'start');
                }
            } else {
                errorMessage = getApiErrorMessage(err, 'start');
            }
            
            dispatch({ type: 'SET_ERROR', payload: errorMessage });
            cleanupAudio();
        }
    }, [script, cleanupAudio, language, micGain, speakerVolume]);

    const handleReset = () => {
        cleanupAudio();
        dispatch({ type: 'RESET' });
    };

    const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = async (e) => {
            try {
                const arrayBuffer = e.target?.result as ArrayBuffer;
                const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
                let fullText = '';
                for (let i = 1; i <= pdf.numPages; i++) {
                    const page = await pdf.getPage(i);
                    const textContent = await page.getTextContent();
                    const pageText = textContent.items.map(item => ('str' in item) ? item.str : '').join(' ');
                    fullText += pageText + '\n';
                }
                dispatch({ type: 'SET_STATE', payload: { script: fullText.trim() } });
            } catch (pdfError) {
                console.error('Error parsing PDF:', pdfError);
                dispatch({ type: 'SET_ERROR', payload: 'Failed to parse the PDF file. Please ensure it is a valid PDF.' });
            }
        };
        reader.readAsArrayBuffer(file);
    };
    
    const handleMicGainChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const newGain = parseFloat(e.target.value);
        dispatch({ type: 'SET_STATE', payload: { micGain: newGain } });
        if (inputGainNode.current && inputAudioContext.current) {
            inputGainNode.current.gain.setValueAtTime(newGain, inputAudioContext.current.currentTime);
        }
    };

    const handleSpeakerVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const newVolume = parseFloat(e.target.value);
        dispatch({ type: 'SET_STATE', payload: { speakerVolume: newVolume } });
        if (outputGainNode.current && outputAudioContext.current) {
            outputGainNode.current.gain.setValueAtTime(newVolume, outputAudioContext.current.currentTime);
        }
    };

    useEffect(() => {
      return () => {
        cleanupAudio();
      };
    }, [cleanupAudio]);

    const renderContent = () => {
        switch (interviewState) {
            case InterviewState.IDLE:
            case InterviewState.ERROR:
                return (
                    <div className="w-full max-w-2xl mx-auto">
                        <h1 className="text-4xl font-bold text-center mb-2">AI Interview Simulator</h1>
                        <p className="text-center text-gray-400 mb-8">Enter a topic or paste a script. The AI will conduct a live audio interview based on your input.</p>
                        
                        <div className="mb-4">
                            <label htmlFor="language-select" className="block text-sm font-medium text-gray-300 mb-2">Interview Language</label>
                            <select
                                id="language-select"
                                value={language}
                                onChange={(e) => dispatch({ type: 'SET_STATE', payload: { language: e.target.value } })}
                                className="w-full p-3 bg-gray-800 border-2 border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all"
                            >
                                {languageOptions.map(option => (
                                    <option key={option.value} value={option.value}>{option.label}</option>
                                ))}
                            </select>
                        </div>

                        <textarea
                            className="w-full h-48 p-4 bg-gray-800 border-2 border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all"
                            value={script}
                            onChange={(e) => dispatch({ type: 'SET_STATE', payload: { script: e.target.value } })}
                            placeholder="e.g., Conduct an interview for a junior software engineer."
                        />
                         <div className="my-6">
                            <AudioControls
                                micGain={micGain}
                                onMicGainChange={handleMicGainChange}
                                speakerVolume={speakerVolume}
                                onSpeakerVolumeChange={handleSpeakerVolumeChange}
                            />
                        </div>
                        {error && <p className="text-red-400 text-center">{error}</p>}
                        <div className="flex flex-col sm:flex-row gap-4 mt-6">
                            <button
                                onClick={() => fileInputRef.current?.click()}
                                className="w-full bg-gray-700 hover:bg-gray-600 text-white font-bold py-3 px-4 rounded-lg flex items-center justify-center text-lg transition-colors"
                            >
                                <UploadIcon className="h-6 w-6 mr-2" />
                                Upload PDF
                            </button>
                             <input type="file" ref={fileInputRef} onChange={handleFileChange} accept=".pdf" className="hidden" />
                            <button
                                onClick={handleStartConversation}
                                className="w-full bg-purple-600 hover:bg-purple-700 text-white font-bold py-3 px-4 rounded-lg flex items-center justify-center text-lg transition-transform transform hover:scale-105"
                            >
                                <PlayIcon className="h-6 w-6 mr-2" />
                                Start Interview
                            </button>
                        </div>
                    </div>
                );
            case InterviewState.CONNECTING:
            case InterviewState.SUMMARIZING:
                 return (
                    <div className="flex flex-col items-center justify-center text-center">
                        <RobotIcon className="h-24 w-24 text-purple-400 animate-pulse" />
                        <h1 className="text-3xl font-bold mt-4">{interviewState === 'CONNECTING' ? 'Connecting...' : 'Generating Feedback...'}</h1>
                        <p className="text-gray-400 mt-2">{interviewState === 'CONNECTING' ? 'Preparing the interview session.' : 'Our AI is analyzing your conversation.'}</p>
                    </div>
                );
            case InterviewState.FINISHED:
                 return (
                    <div className="w-full max-w-4xl mx-auto">
                        <h1 className="text-4xl font-bold text-center mb-6">Interview Complete</h1>
                        <div className="bg-gray-800 p-6 rounded-lg shadow-lg mb-8">
                             <h2 className="text-2xl font-bold text-purple-400 mb-4">Performance Summary</h2>
                             <div className="text-gray-300 whitespace-pre-wrap space-y-2 prose prose-invert prose-p:my-2 prose-headings:my-4">
                                {summary ? <ReactMarkdown remarkPlugins={[remarkGfm]}>{summary}</ReactMarkdown> : "No summary available."}
                             </div>
                        </div>
                         <div className="bg-gray-800 p-6 rounded-lg shadow-lg space-y-4 max-h-[40vh] overflow-y-auto">
                            <h2 className="text-2xl font-bold text-purple-400 mb-4">Full Transcript</h2>
                            {transcript.map((entry, i) => (
                                <div key={i} className={`flex items-start gap-3 ${entry.speaker === 'You' ? 'justify-end' : ''}`}>
                                    {entry.speaker === 'Interviewer' && <RobotIcon className="h-8 w-8 p-1.5 bg-gray-700 text-purple-400 rounded-full flex-shrink-0" />}
                                    <p className={`p-3 rounded-lg max-w-xl ${entry.speaker === 'You' ? 'bg-cyan-800 text-white' : 'bg-gray-700 text-gray-200'}`}>{entry.text}</p>
                                    {entry.speaker === 'You' && <UserIcon className="h-8 w-8 p-1.5 bg-gray-700 text-cyan-400 rounded-full flex-shrink-0" />}
                                </div>
                            ))}
                        </div>
                        <button
                            onClick={handleReset}
                            className="w-full mt-8 bg-purple-600 hover:bg-purple-700 text-white font-bold py-3 px-4 rounded-lg flex items-center justify-center text-lg transition-transform transform hover:scale-105"
                        >
                            Start New Interview
                        </button>
                    </div>
                );
            case InterviewState.IN_CONVERSATION:
                return (
                     <div className="w-full max-w-3xl mx-auto flex flex-col h-[80vh]">
                        <div className="flex-grow overflow-y-auto p-4 space-y-4">
                            {transcript.length === 0 && !currentInterviewerText && !currentUserText && (
                                <div className="flex flex-col items-center justify-center text-center text-gray-400 h-full">
                                    <RobotIcon className="h-16 w-16 mb-4 text-purple-400 animate-pulse" />
                                    <p className="text-lg">The interviewer will begin shortly...</p>
                                </div>
                            )}
                            {transcript.map((entry, i) => (
                                <div key={i} className={`flex items-start gap-3 ${entry.speaker === 'You' ? 'justify-end' : 'justify-start'}`}>
                                    {entry.speaker === 'Interviewer' && <RobotIcon className="h-8 w-8 p-1.5 bg-gray-700 text-purple-400 rounded-full flex-shrink-0 mt-1" />}
                                    <div className={`p-3 rounded-lg max-w-xl ${entry.speaker === 'You' ? 'bg-cyan-800 text-white' : 'bg-gray-700 text-gray-200'}`}>{entry.text}</div>
                                    {entry.speaker === 'You' && <UserIcon className="h-8 w-8 p-1.5 bg-gray-700 text-cyan-400 rounded-full flex-shrink-0 mt-1" />}
                                </div>
                            ))}
                            {currentInterviewerText && (
                                <div className="flex items-start gap-3 justify-start">
                                    <RobotIcon className="h-8 w-8 p-1.5 bg-gray-700 text-purple-400 rounded-full flex-shrink-0 mt-1 animate-pulse" />
                                    <div className="p-3 rounded-lg max-w-xl bg-gray-700 text-gray-200">{currentInterviewerText}</div>
                                </div>
                            )}
                            {currentUserText && (
                                <div className="flex items-start gap-3 justify-end">
                                    <div className="p-3 rounded-lg max-w-xl bg-cyan-800 text-white">{currentUserText}</div>
                                    <UserIcon className="h-8 w-8 p-1.5 bg-gray-700 text-cyan-400 rounded-full flex-shrink-0 mt-1 animate-pulse" />
                                </div>
                            )}
                            <div ref={transcriptEndRef} />
                        </div>
                        <div className="flex-shrink-0 pt-6">
                            <div className="max-w-md mx-auto">
                                <AudioControls
                                    micGain={micGain}
                                    onMicGainChange={handleMicGainChange}
                                    speakerVolume={speakerVolume}
                                    onSpeakerVolumeChange={handleSpeakerVolumeChange}
                                />
                            </div>
                           <div className="flex justify-center mt-6">
                                <button
                                    onClick={handleStopConversation}
                                    className="bg-red-600 hover:bg-red-700 text-white font-bold py-3 px-8 rounded-lg flex items-center justify-center text-lg transition-transform transform hover:scale-105"
                                >
                                    <StopIcon className="h-6 w-6 mr-2" />
                                    End Interview
                                </button>
                            </div>
                        </div>
                    </div>
                );
        }
    };

    return (
        <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center p-4 sm:p-6 lg:p-8">
            <div className="w-full">
                {renderContent()}
            </div>
        </div>
    );
};

export default App;
