//
//  AudioRecorderViewModel.swift
//  ios-app
//
//  Created by Ridhwik Kalgaonkar on 11/8/24.
//

import AVFoundation
import SwiftUI

class AudioRecorderViewModel: NSObject, ObservableObject {
    @Published var isRecording = false
    @Published var recordings: [URL] = []
    @Published var currentPower: Double = 0.0
    
    var audioRecorder: AVAudioRecorder?
    var audioPlayer: AVAudioPlayer?
    private var timer: Timer?
    
    override init() {
        super.init()
        fetchRecordings()
    }
    
    func startRecording() {
        let recordingSession = AVAudioSession.sharedInstance()
        
        do {
            try recordingSession.setCategory(.playAndRecord, mode: .default)
            try recordingSession.setActive(true)
            
            let documentPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let audioFilename = documentPath.appendingPathComponent("\(Date().toString()).m4a")
            
            let settings = [
                AVFormatIDKey: Int(kAudioFormatMPEG4AAC),
                AVSampleRateKey: 44100,
                AVNumberOfChannelsKey: 1,
                AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
            ]
            
            audioRecorder = try AVAudioRecorder(url: audioFilename, settings: settings)
            audioRecorder?.isMeteringEnabled = true
            audioRecorder?.record()
            isRecording = true
            
            startMonitoring()
        } catch {
            print("Could not start recording: \(error)")
        }
    }
    
    func stopRecording() {
        audioRecorder?.stop()
        isRecording = false
        stopMonitoring()
        fetchRecordings()
    }
    
    func startPlayback(audio: URL) {
        do {
            audioPlayer = try AVAudioPlayer(contentsOf: audio)
            audioPlayer?.isMeteringEnabled = true
            audioPlayer?.play()
            startMonitoring()
        } catch {
            print("Playback failed: \(error)")
        }
    }
    
    func stopPlayback() {
        audioPlayer?.stop()
        stopMonitoring()
        currentPower = 0.0
    }
    
    private func startMonitoring() {
        timer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            self?.updatePower()
        }
    }
    
    private func stopMonitoring() {
        timer?.invalidate()
        timer = nil
    }
    
    private func updatePower() {
        if isRecording {
            audioRecorder?.updateMeters()
            let power = audioRecorder?.averagePower(forChannel: 0) ?? -160
            updateCurrentPower(power)
        } else if audioPlayer?.isPlaying == true {
            audioPlayer?.updateMeters()
            let power = audioPlayer?.averagePower(forChannel: 0) ?? -160
            updateCurrentPower(power)
        }
    }
    
    private func updateCurrentPower(_ power: Float) {
        // Convert decibels to linear scale (0-1)
        let minDb: Float = -60
        
        if power < minDb {
            DispatchQueue.main.async {
                self.currentPower = 0.0
            }
            return
        }
        
        // Normalize the value between 0 and 1 and convert to Double
        let normalizedValue = Double(pow(10, (power - minDb) / 20))
        DispatchQueue.main.async {
            self.currentPower = min(max(normalizedValue, 0), 1)
        }
    }
    
    private func fetchRecordings() {
        let documentPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        do {
            let urls = try FileManager.default.contentsOfDirectory(at: documentPath, includingPropertiesForKeys: nil)
            recordings = urls.filter { $0.pathExtension == "m4a" }
                .sorted(by: { $0.lastPathComponent > $1.lastPathComponent })
        } catch {
            print("Could not fetch recordings")
        }
    }
}
