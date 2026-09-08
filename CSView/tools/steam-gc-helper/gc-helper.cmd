@echo off
rem Runs the helper regardless of the current directory, so that cs2mv can be
rem pointed at one path with nothing to quote.
node "%~dp0gc-helper.js" %*
