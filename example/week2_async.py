import asyncio

async def do_some_processing() -> str:
    # do some work
    return "Done"

async def main():
    result = await do_some_processing()
    print(result)

asyncio.run(main())