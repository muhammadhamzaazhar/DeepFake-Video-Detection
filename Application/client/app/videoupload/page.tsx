"use client";

import type React from "react";

import { useState, useRef, useCallback } from "react";
import {
  ArrowLeft,
  Upload,
  AlertCircle,
  AlertTriangle,
  Play,
  FileVideo,
  Zap,
  CheckCircle,
  Loader2,
  X,
  BarChart3,
  Eye,
  Shield,
} from "lucide-react";
import Link from "next/link";
import { supabase } from "@/lib/supabaseClient";

const BUCKET_NAME = "deepfake-videos";
const FOLDER_PATH = "skintegrityvideos";

export default function VideoUpload() {
  const [video, setVideo] = useState<File | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoPreview, setVideoPreview] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<{
    classification: string;
    confidence: number;
    detectedAreas: string[];
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  const handleVideoChange = (file: File) => {
    if (file) {
      setVideo(file);
      setResult(null);
      setError(null);
      const videoUrl = URL.createObjectURL(file);
      setVideoPreview(videoUrl);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleVideoChange(e.target.files[0]);
    }
  };

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleVideoChange(e.dataTransfer.files[0]);
    }
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!video) {
      setError("Please select a video to upload.");
      return;
    }

    setIsLoading(true);
    setResult(null);
    setError(null);

    try {
      const filePath = `${FOLDER_PATH}/${video.name}`;
      const { error: uploadError } = await supabase.storage
        .from(BUCKET_NAME)
        .upload(filePath, video, {
          cacheControl: "3600",
          upsert: true,
        });

      if (uploadError) {
        console.error(uploadError);
        setError("Failed to upload video to Supabase.");
        return;
      }

      const { data } = supabase.storage
        .from(BUCKET_NAME)
        .getPublicUrl(filePath);

      if (!data?.publicUrl) {
        setError("Failed to retrieve video URL.");
        return;
      }
      setVideoUrl(data.publicUrl);

      const API_URL =
        process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
      const response = await fetch(`${API_URL}/api/video`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ url: data.publicUrl }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error("Backend response:", errorText);
        setError("Failed to process video. Server responded with an error.");
        return;
      }

      const resultData = await response.json();
      setResult({
        classification: resultData.classification,
        confidence: resultData.confidence,
        detectedAreas: resultData.detectedAreas,
      });
    } catch (error) {
      console.error(error);
      setError("An unexpected error occurred.");
    } finally {
      setIsLoading(false);
      const { data, error } = await supabase.storage
        .from(BUCKET_NAME)
        .remove([`${FOLDER_PATH}/${video.name}`]);

      if (error) {
        console.error("Error deleting video:", error);
        setError("Failed to delete video.");
      } else {
        console.log("Video deleted successfully:", data);
        setVideoUrl(null);
      }
    }
  };

  const resetForm = () => {
    setVideo(null);
    setVideoPreview(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="circuit-pattern fixed inset-0 z-0"></div>

      <div className="relative z-10 container mx-auto px-6 py-12">
        <Link
          href="/"
          className="inline-flex items-center text-gray-600 dark:text-gray-400 hover:text-red-600 mb-8"
        >
          <ArrowLeft className="mr-2" size={16} />
          Back to Home
        </Link>

        <div className="max-w-4xl mx-auto">
          <span className="text-4xl font-bold">
            <span className="text-black">Skin</span>
            <span className="text-red-600">tegrity</span>
          </span>

          <p className="text-gray-600 dark:text-gray-400 mb-8 mt-2">
            Upload a video to analyze for potential deepfake manipulation
          </p>
        </div>

        <div className="max-w-5xl mx-auto">
          {!result ? (
            /* Upload Section */
            <div className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl rounded-3xl shadow-2xl border border-white/20 dark:border-slate-700/50 overflow-hidden">
              <div className="p-6 sm:p-10">
                <form onSubmit={handleSubmit} className="space-y-8">
                  {/* Upload Zone */}
                  <div
                    className={`relative group cursor-pointer transition-all duration-300 ease-out ${
                      isDragging
                        ? "scale-105 border-red-400 bg-red-50/50 dark:bg-red-950/20 shadow-2xl"
                        : videoPreview
                        ? "border-emerald-400 bg-emerald-50/50 dark:bg-emerald-950/20 shadow-xl"
                        : "border-slate-300 dark:border-slate-600 hover:border-red-400 hover:shadow-xl"
                    } border-2 border-dashed rounded-2xl p-8 sm:p-12 text-center`}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                  >
                    {videoPreview ? (
                      <div className="w-full space-y-6">
                        {/* Video Preview */}
                        <div className="relative aspect-video w-full max-w-2xl mx-auto">
                          <video
                            ref={videoRef}
                            src={videoPreview}
                            className="w-full h-full rounded-xl object-cover shadow-lg border border-slate-200 dark:border-slate-700"
                            controls
                          />
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              resetForm();
                            }}
                            className="absolute -top-3 -right-3 bg-red-500 hover:bg-red-600 text-white p-2 rounded-full shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-110"
                          >
                            <X size={16} />
                          </button>
                          <div className="absolute bottom-4 left-4 bg-black/70 backdrop-blur-sm text-white px-3 py-1 rounded-full text-sm flex items-center space-x-2">
                            <Play size={14} />
                            <span>Ready to analyze</span>
                          </div>
                        </div>

                        {/* File Info */}
                        <div className="bg-white/60 dark:bg-slate-800/60 backdrop-blur-sm rounded-xl p-4 border border-slate-200/50 dark:border-slate-700/50">
                          <div className="flex items-center justify-center space-x-3">
                            <FileVideo className="text-emerald-600" size={20} />
                            <span className="font-medium text-slate-700 dark:text-slate-300">
                              {video?.name}
                            </span>
                            <span className="text-sm text-slate-500">
                              (
                              {(video?.size || 0) / (1024 * 1024) < 1
                                ? `${((video?.size || 0) / 1024).toFixed(0)} KB`
                                : `${(
                                    (video?.size || 0) /
                                    (1024 * 1024)
                                  ).toFixed(1)} MB`}
                              )
                            </span>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-6">
                        <div className="flex justify-center">
                          <div className="relative">
                            <div className="w-20 h-20 bg-gradient-to-br from-red-500 to-rose-600 rounded-full flex items-center justify-center shadow-lg">
                              <Upload className="w-10 h-10 text-white" />
                            </div>
                            <div className="absolute -top-1 -right-1 w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center">
                              <Zap className="w-3 h-3 text-white" />
                            </div>
                          </div>
                        </div>

                        <div className="space-y-3">
                          <h3 className="text-xl font-semibold text-slate-800 dark:text-slate-200">
                            Drop your video here to get started
                          </h3>
                          <p className="text-slate-600 dark:text-slate-400">
                            or click to browse your files
                          </p>
                        </div>

                        <div className="inline-flex items-center space-x-4 text-sm">
                          <span className="bg-slate-200/50 dark:bg-slate-800/50 px-4 py-2 rounded-full text-slate-500 dark:text-slate-400 font-medium">
                            Supports MP4
                          </span>
                          <span className="bg-slate-200/50 dark:bg-slate-800/50 px-4 py-2 rounded-full text-slate-500 dark:text-slate-400 font-semibold">
                            Up to 100MB
                          </span>
                        </div>
                      </div>
                    )}

                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="video/*"
                      onChange={handleFileInputChange}
                      className="hidden"
                    />
                  </div>

                  {/* Error Message */}
                  {error && (
                    <div className="bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800/50 rounded-xl p-4 flex items-start space-x-3">
                      <AlertTriangle
                        className="text-red-500 mt-0.5 flex-shrink-0"
                        size={20}
                      />
                      <p className="text-red-700 dark:text-red-400 font-medium">
                        {error}
                      </p>
                    </div>
                  )}

                  {/* Submit Button */}
                  <div className="flex justify-center">
                    <button
                      type="submit"
                      className={`relative group px-8 py-4 border-2 border-rose-500 text-rose-500 
                      bg-transparent font-semibold rounded-xl transition-all duration-300 
                      disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-3 
                      min-w-[220px] justify-center 
                      ${
                        isLoading
                          ? "animate-pulse"
                          : "hover:bg-rose-500 hover:text-white hover:border-rose-500 hover:scale-105"
                      }`}
                      disabled={isLoading || !video}
                    >
                      {isLoading ? (
                        <>
                          <Loader2 className="animate-spin" size={20} />
                          <span>Analyzing Video...</span>
                        </>
                      ) : (
                        <>
                          <Eye size={20} />
                          <span>Analyze Video</span>
                        </>
                      )}
                    </button>
                  </div>
                </form>
              </div>
            </div>
          ) : (
            /* Results Section */
            <div className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl rounded-3xl shadow-2xl border border-white/20 dark:border-slate-700/50 overflow-hidden">
              <div className="p-6 sm:p-10">
                <div className="grid lg:grid-cols-2 gap-8 lg:gap-12">
                  {/* Video Section */}
                  <div className="space-y-6">
                    {videoPreview && (
                      <div className="aspect-video w-full">
                        <video
                          src={videoPreview}
                          className="w-full h-full rounded-xl object-cover shadow-lg border border-slate-200 dark:border-slate-700"
                          controls
                        />
                      </div>
                    )}

                    <div className="bg-slate-50/50 dark:bg-slate-800/50 rounded-xl p-6 space-y-3">
                      <div className="flex items-center space-x-3 text-slate-600 dark:text-slate-400">
                        <FileVideo size={18} />
                        <span className="font-medium">{video?.name}</span>
                      </div>
                      <div className="text-sm text-slate-500 dark:text-slate-500">
                        Size:{" "}
                        {(video?.size || 0) / (1024 * 1024) < 1
                          ? `${((video?.size || 0) / 1024).toFixed(0)} KB`
                          : `${((video?.size || 0) / (1024 * 1024)).toFixed(
                              1
                            )} MB`}
                      </div>
                    </div>
                  </div>

                  {/* Results Section */}
                  <div className="space-y-6">
                    <div
                      className={`relative overflow-hidden rounded-2xl p-8 ${
                        result.classification === "REAL"
                          ? "bg-gradient-to-br from-emerald-50 to-green-50 dark:from-emerald-950/30 dark:to-green-950/30 border-2 border-emerald-200 dark:border-emerald-800/50"
                          : "bg-gradient-to-br from-red-50 to-rose-50 dark:from-red-950/30 dark:to-rose-950/30 border-2 border-red-200 dark:border-red-800/50"
                      }`}
                    >
                      {/* Background Pattern */}
                      <div className="absolute inset-0 opacity-5">
                        <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_40%,rgba(0,0,0,0.1),transparent_50%)]"></div>
                      </div>

                      <div className="relative space-y-6">
                        {/* Result Header */}
                        <div className="flex items-center space-x-4">
                          <div
                            className={`p-3 rounded-full ${
                              result.classification === "REAL"
                                ? "bg-emerald-100 dark:bg-emerald-900/50"
                                : "bg-red-100 dark:bg-red-900/50"
                            }`}
                          >
                            {result.classification === "REAL" ? (
                              <Shield
                                className="text-emerald-600 dark:text-emerald-400"
                                size={28}
                              />
                            ) : (
                              <AlertTriangle
                                className="text-red-600 dark:text-red-400"
                                size={28}
                              />
                            )}
                          </div>

                          <div>
                            <h2 className="text-2xl font-bold text-slate-800 dark:text-slate-200">
                              {result.classification === "REAL"
                                ? "Authentic Video"
                                : "Deepfake Detected"}
                            </h2>
                            <p
                              className={`text-sm font-medium ${
                                result.classification === "REAL"
                                  ? "text-emerald-700 dark:text-emerald-300"
                                  : "text-red-700 dark:text-red-300"
                              }`}
                            >
                              {result.classification === "REAL"
                                ? "This video appears to be genuine"
                                : "Artificial manipulation detected"}
                            </p>
                          </div>
                        </div>

                        {/* Confidence Score */}
                        <div className="space-y-3">
                          <div className="flex items-center justify-between">
                            <span className="text-slate-700 dark:text-slate-300 font-medium flex items-center space-x-2">
                              <BarChart3 size={18} />
                              <span>Confidence Score</span>
                            </span>
                            <span className="text-2xl font-bold text-slate-800 dark:text-slate-200">
                              {result.confidence.toFixed(1)}%
                            </span>
                          </div>

                          <div className="relative">
                            <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-3 overflow-hidden">
                              <div
                                className={`h-full rounded-full transition-all duration-1000 ease-out ${
                                  result.classification === "REAL"
                                    ? "bg-gradient-to-r from-emerald-500 to-green-500"
                                    : "bg-gradient-to-r from-red-500 to-rose-500"
                                } shadow-lg`}
                                style={{
                                  width: `${result.confidence}%`,
                                }}
                              ></div>
                            </div>
                            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-pulse"></div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Action Button */}
                    <div className="mt-6 flex flex-col sm:flex-row gap-4">
                      <button
                        onClick={resetForm}
                        className="w-full btn-outline text-gray-700 font-semibold rounded-xl border border-dashed border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500 bg-white dark:bg-gray-800 flex items-center justify-center space-x-2 py-2 px-4 transition-colors"
                      >
                        <Upload
                          size={16}
                          className="text-gray-500 dark:text-gray-400"
                        />
                        <span>Analyze Another Video</span>
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className="mt-16 space-y-8">
            <div className="text-center">
              <h2 className="text-3xl font-bold text-slate-800 dark:text-slate-200 mb-4">
                How It Works
              </h2>
              <p className="text-lg text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
                Our advanced AI technology uses multiple detection layers to
                identify deepfake content
              </p>
            </div>

            <div className="grid md:grid-cols-3 gap-8">
              {[
                {
                  step: "1",
                  title: "Video Analysis",
                  description:
                    "Our AI scans each frame for inconsistencies in facial features and movement patterns",
                  icon: Eye,
                  color: "from-blue-500 to-cyan-500",
                },
                {
                  step: "2",
                  title: "Pattern Recognition",
                  description:
                    "Advanced algorithms detect unnatural patterns in facial blood flow and texture mapping",
                  icon: BarChart3,
                  color: "from-purple-500 to-pink-500",
                },
                {
                  step: "3",
                  title: "Result Generation",
                  description:
                    "Provides a confidence score and a clear authenticity verdict to determine if the video is genuine or manipulated",
                  icon: Shield,
                  color: "from-emerald-500 to-green-500",
                },
              ].map((item, index) => {
                const IconComponent = item.icon;
                return (
                  <div
                    key={index}
                    className="group relative bg-white/60 dark:bg-slate-800/60 backdrop-blur-sm rounded-2xl p-8 border border-slate-200/50 dark:border-slate-700/50 hover:shadow-xl transition-all duration-300 hover:-translate-y-2"
                  >
                    <div className="space-y-6 text-center">
                      <div className="relative mx-auto w-fit">
                        <div
                          className={`w-16 h-16 bg-gradient-to-br ${item.color} rounded-2xl flex items-center justify-center shadow-lg group-hover:shadow-xl transition-shadow duration-300`}
                        >
                          <IconComponent className="text-white" size={28} />
                        </div>
                        <div className="absolute -bottom-2 -right-2 w-8 h-8 bg-slate-800 dark:bg-slate-200 text-white dark:text-slate-800 rounded-full flex items-center justify-center text-sm font-bold shadow-lg">
                          {item.step}
                        </div>
                      </div>

                      <div className="space-y-3">
                        <h3 className="text-xl font-semibold text-slate-800 dark:text-slate-200">
                          {item.title}
                        </h3>
                        <p className="text-slate-600 dark:text-slate-400 leading-relaxed">
                          {item.description}
                        </p>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
