
import { GoogleGenAI, Modality } from "@google/genai";
import { TranscriptEntry } from "../types";
import { getApiErrorMessage } from "../utils/error";

if (!process.env.API_KEY) {
    throw new Error("API_KEY environment variable is not set");
}

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

export const generateWelcomeAudio = async (language: string): Promise<string> => {
    const welcomeMessages: { [key: string]: string } = {
        'English': 'Hello and welcome to your interview. The session will begin shortly.',
        'Spanish': 'Hola y bienvenido a tu entrevista. La sesión comenzará en breve.',
        'French': 'Bonjour et bienvenue à votre entretien. La session va bientôt commencer.',
        'German': 'Hallo und herzlich willkommen zu Ihrem Interview. Die Sitzung beginnt in Kürze.',
        'Japanese': 'こんにちは、面接へようこそ。セッションは間もなく開始されます。',
        'Mandarin Chinese': '您好，欢迎参加面试。面试很快就会开始。',
        'Hindi': 'नमस्ते और आपके साक्षात्कार में आपका स्वागत है। सत्र शीघ्र ही शुरू होगा।',
        'Portuguese': 'Olá e bem-vindo à sua entrevista. A sessão começará em breve.',
    };

    const text = welcomeMessages[language] || welcomeMessages['English'];
    
    try {
        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash-preview-tts",
            contents: [{ parts: [{ text }] }],
            config: {
                responseModalities: [Modality.AUDIO],
                speechConfig: {
                    voiceConfig: {
                        prebuiltVoiceConfig: { voiceName: 'Kore' }, // Use the same voice for consistency
                    },
                },
            },
        });

        const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
        if (!base64Audio) {
            console.warn("TTS API did not return audio data for welcome message.");
            return '';
        }
        return base64Audio;
    } catch (error) {
        console.error("Error generating welcome audio:", error);
        // Fail gracefully so interview can still start
        return '';
    }
};


export const generateInterviewSummary = async (transcript: TranscriptEntry[], language: string): Promise<string> => {
    const formattedTranscript = transcript.map(entry => `${entry.speaker}: ${entry.text}`).join('\n\n');

    const prompt = `You are an expert HR manager providing feedback on a job interview.
Provide the feedback in ${language}.

Based on the following interview transcript, please provide a comprehensive summary of the candidate's performance.

Your feedback should include:
1.  **Overall Summary:** A brief, high-level overview of the interview.
2.  **Strengths:** Identify specific examples from the candidate's answers that demonstrate their strengths.
3.  **Areas for Improvement:** Offer constructive feedback on where the candidate could improve, again, referencing their answers.
4.  **Final Recommendation:** A concluding thought on the candidate's suitability for a generic role based on this interview.

Format your response clearly using markdown headings.

**Interview Transcript:**
---
${formattedTranscript}
---
`;

    try {
        const response = await ai.models.generateContent({
            model: 'gemini-2.5-pro',
            contents: prompt,
        });
        return response.text;
    } catch (error) {
        console.error("Error generating summary:", error);
        throw new Error(getApiErrorMessage(error, 'summary'));
    }
};