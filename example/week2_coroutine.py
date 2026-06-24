import asyncio

async def do_some_processing():
    print("Processing started...")
    await asyncio.sleep(2)  # Non-blocking wait
    print("Processing completed...")
    return "Done"

async def main():
    # Create coroutine object
    coroutine = do_some_processing()
    print(coroutine)

    # Execute coroutine
    result = await coroutine
    print("Result:", result)

asyncio.run(main())