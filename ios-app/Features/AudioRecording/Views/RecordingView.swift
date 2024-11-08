//
//  RecordingView.swift
//  ios-app
//
//  Created by Ridhwik Kalgaonkar on 11/8/24.
//

import SwiftUI
import SiriWaveView
import AVFoundation

struct RecordingView: View {
    @StateObject private var recorder = AudioRecorderViewModel()
    @State private var apiResponse: String = ""
    @State private var isLoading: Bool = false
    
    var body: some View {
        NavigationView {
            VStack {
                // Siri Waveform visualization
                SiriWaveView(power: $recorder.currentPower)
                    .opacity(0.5)
                    .frame(height: 100)
                    .padding()
                    .background(Color.black.opacity(0.1))
                    .cornerRadius(10)
                
                Button(action: {
                    if recorder.isRecording {
                        recorder.stopRecording()
                    } else {
                        recorder.startRecording()
                    }
                }) {
                    Image(systemName: recorder.isRecording ? "stop.circle.fill" : "mic.circle.fill")
                        .font(.system(size: 64))
                        .foregroundColor(recorder.isRecording ? .red : .blue)
                }
                
                VStack {
                                    Button(action: {
                                        Task {
                                            await fetchHelloMessage()
                                        }
                                    }) {
                                        HStack {
                                            Text("Test API")
                                            if isLoading {
                                                ProgressView()
                                                    .progressViewStyle(CircularProgressViewStyle())
                                            }
                                        }
                                        .frame(maxWidth: .infinity)
                                        .padding()
                                        .background(Color.blue)
                                        .foregroundColor(.white)
                                        .cornerRadius(10)
                                    }
                                    .disabled(isLoading)
                                    
                                    if !apiResponse.isEmpty {
                                        Text(apiResponse)
                                            .padding()
                                            .background(Color.gray.opacity(0.1))
                                            .cornerRadius(8)
                                    }
                                }
                                .padding()
                
                List {
                    ForEach(recorder.recordings, id: \.self) { recording in
                        Button(action: {
                            if recorder.audioPlayer?.isPlaying == true {
                                recorder.stopPlayback()
                            } else {
                                recorder.startPlayback(audio: recording)
                            }
                        }) {
                            HStack {
                                Image(systemName: "play.circle.fill")
                                    .foregroundColor(recorder.audioPlayer?.isPlaying == true ? .red : .blue)
                                Text(recording.lastPathComponent)
                            }
                        }
                    }
                }
            }
            .navigationTitle("Voice Recorder")
        }
    }
    
    func fetchHelloMessage() async {
            isLoading = true
            defer { isLoading = false }
            
            guard let url = URL(string: "http://127.0.0.1:5000/api/hello") else {
                apiResponse = "Invalid URL"
                return
            }
            
            do {
                let (data, _) = try await URLSession.shared.data(from: url)
                
                if let response = String(data: data, encoding: .utf8) {
                    await MainActor.run {
                        apiResponse = response
                    }
                }
            } catch {
                await MainActor.run {
                    apiResponse = "Error: \(error.localizedDescription)"
                }
            }
        }
}
