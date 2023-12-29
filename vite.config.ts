import { sveltekit } from "@sveltejs/kit/vite";
import { defineConfig } from "vite";
import Icons from "unplugin-icons/vite";

export default defineConfig({
	
	server: {
		host: '0.0.0.0',
        port: 2160  // Set your desired port here
    },
	plugins: [
		sveltekit(),
		Icons({
			compiler: "svelte",
		}),
	],
});