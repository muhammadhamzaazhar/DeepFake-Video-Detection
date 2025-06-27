"use client";

import { useState } from "react";
import Image from "next/image";
import Link from "next/link";
import {
  ArrowRight,
  Upload,
  Shield,
  AlertTriangle,
  Twitter,
  Linkedin,
  Github,
} from "lucide-react";

export default function Home() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  return (
    <div className="min-h-screen flex flex-col">
      {/* Circuit pattern background */}
      <div className="circuit-pattern fixed inset-0 z-0"></div>

      {/* Navigation */}
      <header className="relative z-10 bg-white/80 dark:bg-gray-900/80 backdrop-blur-md">
        <nav className="container mx-auto px-6 py-4 flex items-center justify-between">
          <Link href="/" className="flex items-center space-x-2">
            <span className="text-2xl font-bold">
              <span className="text-white dark:text-white">Skin</span>
              <span className="text-red-600">tegrity</span>
            </span>
          </Link>

          {/* Mobile menu button */}
          <button
            className="md:hidden text-gray-600 dark:text-gray-300"
            onClick={() => setIsMenuOpen(!isMenuOpen)}
          >
            <svg
              className="w-6 h-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              {isMenuOpen ? (
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              ) : (
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 6h16M4 12h16M4 18h16"
                />
              )}
            </svg>
          </button>

          {/* Desktop navigation */}
          <div className="hidden md:flex items-center space-x-8">
            <Link
              href="/about"
              className="text-gray-600 dark:text-gray-300 hover:text-red-600 transition-colors"
            >
              About
            </Link>
            <Link
              href="/technology"
              className="text-gray-600 dark:text-gray-300 hover:text-red-600 transition-colors"
            >
              Technology
            </Link>
            <Link
              href="/videoupload"
              className="text-gray-600 dark:text-gray-300 hover:text-red-600 transition-colors"
            >
              Scanner
            </Link>
            <Link
              href="/contact"
              className="btn-primary hover:scale-105 transition-transform"
            >
              Contact Us
            </Link>
          </div>
        </nav>

        {/* Mobile menu */}
        {isMenuOpen && (
          <div className="md:hidden bg-white dark:bg-gray-900 py-4 px-6">
            <div className="flex flex-col space-y-4">
              <Link
                href="/about"
                className="text-gray-600 dark:text-gray-300 hover:text-red-600 transition-colors"
              >
                About
              </Link>
              <Link
                href="/technology"
                className="text-gray-600 dark:text-gray-300 hover:text-red-600 transition-colors"
              >
                Technology
              </Link>
              <Link
                href="/videoupload"
                className="text-gray-600 dark:text-gray-300 hover:text-red-600 transition-colors"
              >
                Scanner
              </Link>
              <Link
                href="/contact"
                className="btn-primary inline-block text-center hover:scale-105 transition-transform"
              >
                Contact Us
              </Link>
            </div>
          </div>
        )}
      </header>

      {/* Hero Section */}
      <section className="relative z-10 min-h-[500px] md:min-h-screen max-h-screen container mx-auto px-6 py-12 md:py-0 flex flex-col md:flex-row items-center justify-center overflow-hidden">
        <div className="md:w-1/2 mb-8 md:mb-0">
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold mb-4 md:mb-6 animate-fade-in">
            Detect <span className="gradient-text">Deepfakes</span> with
            Skintegrity
          </h1>
          <p className="text-lg md:text-xl mb-6 md:mb-8 text-gray-500 dark:text-gray-300 max-w-lg animate-fade-in-delay">
            Our cutting-edge technology analyzes facial patterns and digital
            artifacts to identify synthetically manipulated videos.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 animate-fade-in-delay-2">
            <Link
              href="/videoupload"
              className="btn-primary text-center hover:scale-105 transition-transform"
            >
              Try Scanner Now
            </Link>
            <Link
              href="/technology"
              className="btn-outline text-center hover:scale-105 hover:text-red-600 transition-all duration-300"
            >
              Learn More
            </Link>
          </div>
        </div>
        <div className="md:w-1/2 relative flex justify-center items-center animate-fade-in-right">
          <div className="relative w-full flex justify-center">
            <Image
              src="/images/facial-analysis.png"
              alt="AI facial analysis visualization"
              width={300}
              height={300}
              className="rounded-lg shadow-2xl object-contain md:max-h-[70vh]"
              priority
            />
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="relative z-10 bg-white/80 dark:bg-gray-900/80 backdrop-blur-md py-16">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl md:text-4xl font-bold text-center mb-12 text-gray-900 dark:text-white">
            How Skintegrity Works
          </h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="bg-gray-50 dark:bg-gray-800 p-6 rounded-lg hover:-translate-y-1 hover:shadow-lg transition-all duration-300">
              <div className="bg-red-100 dark:bg-red-900/30 p-4 rounded-full w-16 h-16 flex items-center justify-center mb-4">
                <Upload className="text-red-600" size={32} />
              </div>
              <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                Upload Video
              </h3>
              <p className="text-gray-600 dark:text-gray-400">
                Upload any suspicious video file to our secure platform for
                analysis.
              </p>
            </div>
            <div className="bg-gray-50 dark:bg-gray-800 p-6 rounded-lg hover:-translate-y-1 hover:shadow-lg transition-all duration-300">
              <div className="bg-red-100 dark:bg-red-900/30 p-4 rounded-full w-16 h-16 flex items-center justify-center mb-4">
                <Shield className="text-red-600" size={32} />
              </div>
              <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                AI Analysis
              </h3>
              <p className="text-gray-600 dark:text-gray-400">
                Our advanced AI scans for digital artifacts and inconsistencies
                in facial patterns.
              </p>
            </div>
            <div className="bg-gray-50 dark:bg-gray-800 p-6 rounded-lg hover:-translate-y-1 hover:shadow-lg transition-all duration-300">
              <div className="bg-red-100 dark:bg-red-900/30 p-4 rounded-full w-16 h-16 flex items-center justify-center mb-4">
                <AlertTriangle className="text-red-600" size={32} />
              </div>
              <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                Get Results
              </h3>
              <p className="text-gray-600 dark:text-gray-400">
                Receive a detailed report with confidence scores and highlighted
                areas of concern.
              </p>
            </div>
          </div>
          <div className="mt-12 text-center">
            <Link
              href="/videoupload"
              className="inline-flex items-center btn-primary hover:scale-105 transition-transform"
            >
              Try Scanner Now <ArrowRight className="ml-2" size={16} />
            </Link>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative z-10 py-16 md:py-24">
        <div className="container mx-auto px-6">
          <div className="bg-gradient-to-r from-gray-900 to-gray-800 rounded-2xl p-8 md:p-12 shadow-xl">
            <div className="flex flex-col md:flex-row items-center">
              <div className="md:w-2/3 mb-8 md:mb-0">
                <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
                  Protect Your Organization from Deepfake Threats
                </h2>
                <p className="text-lg text-gray-300 mb-6 max-w-2xl">
                  Deepfakes pose a significant security risk to organizations
                  worldwide. Our enterprise solutions provide continuous
                  monitoring and protection.
                </p>
                <Link
                  href="/contact"
                  className="inline-block bg-red-600 hover:bg-red-700 text-white font-medium py-3 px-8 rounded-md transition-all duration-300 hover:scale-105"
                >
                  Contact for Enterprise Solutions
                </Link>
              </div>
              <div className="md:w-1/3 flex justify-center">
                <div className="w-32 h-32 md:w-48 md:h-48 bg-red-500/20 rounded-full flex items-center justify-center">
                  <Shield className="text-red-500 w-16 h-16 md:w-24 md:h-24" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 bg-white/80 dark:bg-gray-900/80 backdrop-blur-md py-8">
        <div className="container mx-auto px-6">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="mb-4 md:mb-0 text-center md:text-left">
              <span className="text-xl font-bold">
                Skin<span className="text-red-600">tegrity</span>
              </span>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                © {new Date().getFullYear()} Skintegrity. All rights reserved.
              </p>
            </div>
            <div className="flex flex-col md:flex-row items-center space-y-4 md:space-y-0 md:space-x-8">
              <div className="flex space-x-6">
                <Link
                  href="/privacy"
                  className="text-sm text-gray-600 dark:text-gray-400 hover:text-red-600 hover:underline transition-all"
                >
                  Privacy Policy
                </Link>
                <Link
                  href="/terms"
                  className="text-sm text-gray-600 dark:text-gray-400 hover:text-red-600 hover:underline transition-all"
                >
                  Terms of Service
                </Link>
                <Link
                  href="/contact"
                  className="text-sm text-gray-600 dark:text-gray-400 hover:text-red-600 hover:underline transition-all"
                >
                  Contact
                </Link>
              </div>
              <div className="flex space-x-4">
                <a
                  href="https://twitter.com"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Twitter className="w-6 h-6 text-gray-600 dark:text-gray-400 hover:text-red-600 transition-colors" />
                </a>
                <a
                  href="https://linkedin.com"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Linkedin className="w-6 h-6 text-gray-600 dark:text-gray-400 hover:text-red-600 transition-colors" />
                </a>
                <a
                  href="https://github.com"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Github className="w-6 h-6 text-gray-600 dark:text-gray-400 hover:text-red-600 transition-colors" />
                </a>
              </div>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
