using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Audio;

[RequireComponent(typeof(AudioSource))]
public class AudioTest : MonoBehaviour
{

    public int samples = 8192;
    public FFTWindow fftWindow;
    public float minThreshold = 0.02f;
    public float frequency = 0.0f;
    AudioSource _audioSource;
    
    // Start is called before the first frame update
    void Start() {
        _audioSource = GetComponent<AudioSource>();


        _audioSource.Stop();
        _audioSource.clip = Microphone.Start(null, true, 10, AudioSettings.outputSampleRate);
        _audioSource.loop = true;
        // Mute the sound with an Audio Mixer group becuase we don't want the player to hear it
        Debug.Log(Microphone.IsRecording(null).ToString());


        if (Microphone.IsRecording(null))
        { //check that the mic is recording, otherwise you'll get stuck in an infinite loop waiting for it to start
            while (!(Microphone.GetPosition(null) > 0))
            {
            } // Wait until the recording has started. 
            _audioSource.Play();
        }
    }

    // Update is called once per frame
    void Update()
    {
        drawFrecuency(_audioSource);
        Debug.Log("Intensity: " + GetIntensity(_audioSource));
        Debug.Log("Frecuency: " + GetFrecuency(_audioSource));
    }


    public float GetIntensity(AudioSource audioSource)
    {
        float[] spectrum = new float[256];
        float a = 0;
        audioSource.GetOutputData(spectrum, 0);

        foreach (float s in spectrum)
        {
            a += Mathf.Abs(s);
        }
        return a / 256;
    }

    public float GetFrecuency(AudioSource audioSource)
    {
        float fundamentalFrequency = 0.0f;
        float[] spectrum = new float[samples];
        audioSource.GetSpectrumData(spectrum, 0, FFTWindow.BlackmanHarris);

        float s = 0.0f;
        int k = 0;
        for (int j = 1; j < samples; j++)
        {
            if (spectrum[j] > minThreshold) // volumn must meet minimum threshold
            {
                if (s < spectrum[j])
                {
                    s = spectrum[j];
                    k = j;
                }
            }
        }

        fundamentalFrequency = k * AudioSettings.outputSampleRate / samples;
        frequency = fundamentalFrequency;
        return fundamentalFrequency;
    }

    public void drawFrecuency(AudioSource audioSource)
    {
        float[] spectrum = new float[1024];
        audioSource.GetSpectrumData(spectrum, 0, FFTWindow.BlackmanHarris);

        for (int i = 1; i < spectrum.Length - 1; i++)
        {

            float resolution = AudioSettings.outputSampleRate / 2 / 1024;
            float a0, a1;
            a0 = Mathf.Log(spectrum[i - 1]) + 10;
            a1 = Mathf.Log(spectrum[i]) + 10;
            Debug.DrawLine(new Vector3(i - 1, a0, 2), new Vector3(i, a1, 2), Color.cyan);
            Debug.DrawLine(new Vector3(i - 1, Mathf.Log(spectrum[i - 1]), 3), new Vector3(i, Mathf.Log(spectrum[i]), 3), Color.blue);
        }
    }

}
