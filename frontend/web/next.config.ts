import path from "node:path";
import type { NextConfig } from "next";

// Three builds from one config:
// - STATIC_EXPORT: plain HTML/JS/CSS in out/, served by S3 + CloudFront on AWS.
// - DOCKER_BUILD: a standalone Node server for the custom Docker image.
// - neither: Vercel's native deployment.
const output = process.env.STATIC_EXPORT
  ? "export"
  : process.env.DOCKER_BUILD
    ? "standalone"
    : undefined;

const nextConfig: NextConfig = {
  output,
  turbopack: { root: path.resolve(__dirname) },
};

export default nextConfig;
