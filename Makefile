BINARY_NAME=lqai
.DEFAULT_GOAL := run

run:
	odin run .\src\ -out:.\bin\debug\${BINARY_NAME}.exe

runb:
	odin run .\src\${BINARY_NAME}\ -out:.\bin\debug\${BINARY_NAME}.exe